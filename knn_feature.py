### System
import os, sys
from os.path import join
import h5py
import math
from math import floor
import torch
from time import time
from tqdm import tqdm
import argparse

### Numerical Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

### Graph Network Packages
import nmslib
from itertools import chain



#Thanks to open-source implementation of WSI-Graph Construction in PatchGCN: https://github.com/mahmoodlab/Patch-GCN/blob/master/WSI-Graph%20Construction.ipynb
class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=False):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


def pt2graph(wsi_pt, radius=49, ratio=0.2, min_size=200):
    score, coords, features = wsi_pt[:, 0], wsi_pt[:, 1:3], wsi_pt[:, 3:]

    #---->padding
    window_size = radius

    h_ = features.shape[0]
    add_length = (h_//window_size+1)*window_size - h_
    #---->feature
    features = np.pad(array=features, pad_width=((add_length//2, add_length-add_length//2),(0,0)), mode='reflect') #reflect padding
    coords = np.pad(array=coords, pad_width=((add_length//2, add_length-add_length//2),(0,0)), mode='reflect') #reflect padding
    score = np.pad(array=score, pad_width=((add_length//2, add_length-add_length//2)), mode='reflect') #reflect padding


    #---->Processing score
    score_buff = np.sort(score)
    half = len(score_buff) // 2
    score_median = (score_buff[half]+score_buff[~half])/2-0.00000001 

    score_roi = []
    for i in range(len(score)):
        if score[i] > score_median:
            score_roi.append(1)
        else:
            score_roi.append(0)


    num_patches = features.shape[0]
    
    #---->Recombine feature order. Do cluster only one time, it is allowed to have duplicates among different clusters.
    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius)
    b = np.fromiter(chain(*[model.query(features[v_idx], topn=radius) for v_idx in range(num_patches)]),dtype=int)

    features_knn = []
    coords_knn = []
    score_knn = []
    for v_idx in range(0, len(a), window_size):
        select_idx = a[v_idx]
        neighbor_idx = b[v_idx: v_idx+window_size]
        score_idx = [score_roi[idx] for idx in neighbor_idx]
        roi_count = np.where(np.array(score_idx)==1)[0].shape[0]
        if roi_count/window_size > ratio:
            features_knn.append(features[select_idx])
            coords_knn.append(coords[select_idx])
            score_knn.append(score[select_idx])

    score = torch.from_numpy(np.array(score_knn)).unsqueeze(1)
    features = torch.from_numpy(np.stack(features_knn, axis=0)).to(torch.float32)
    coords = torch.from_numpy(np.stack(coords_knn, axis=0))
    #---->concat features and coordinates
    features = torch.cat((score, coords, features), dim=1)

    # #---->Recombine feature order, do cluster several times, it is not allowed to have duplicates among different clusters. 
    # #     This can lead to better results, but at the same time may incur a much higher CPU computational cost.
    # features_knn = []
    # coords_knn = []
    # score_knn = []
    # # max_distance = []
    # for v_idx in tqdm(range(0, num_patches, window_size)):
    #     model = Hnsw(space='l2')
    #     model.fit(features)
    #     select_index = model.query(features[0], topn=radius) #Choose 48 adjacent ones including yourself, a total of 49


    #     score_idx = [score_roi[idx] for idx in select_index]
    #     roi_count = np.where(np.array(score_idx)==1)[0].shape[0]
    #     if roi_count/window_size > ratio:
    #         features_knn.extend(features[select_index])
    #         coords_knn.extend(coords[select_index])
    #         score_knn.append(score[select_index][:, np.newaxis])
    # #         # break #debug




    #     # max_distance.append(np.max((coords[select_index][np.newaxis, :, :]-coords[select_index][:, np.newaxis, :]).sum(-1)))

    #     #---->delete selected features
    #     features = np.delete(features, select_index, axis=0)
    #     coords = np.delete(coords, select_index, axis=0)
    #     score = np.delete(score, select_index, axis=0)

    # score = torch.from_numpy(np.stack(features_knn, axis=0))
    # features = torch.from_numpy(np.stack(features_knn, axis=0)).to(torch.float32)
    # coords = torch.from_numpy(np.stack(coords_knn, axis=0))
    # #---->concat features and coordinates
    # features = torch.cat((score, coords, features), dim=1)


    if features.shape[0] < min_size: #If you can't find the right one, you will not handle it
        return wsi_pt 
    return features

def createDir_pttoPyG(pt_path, save_path, radius, min_size, ratio):
    pbar = tqdm(os.listdir(pt_path))
    for pt_fname in pbar:
        pbar.set_description('%s - Creating Graph' % (pt_fname))
        try:
            if os.path.exists(os.path.join(save_path, pt_fname[:-3]+'.pt')):
                if torch.load(os.path.join(save_path, pt_fname)).shape[0] > min_size:
                    continue
            wsi_pt = torch.load(os.path.join(pt_path, pt_fname))
            G = pt2graph(wsi_pt, radius, ratio, min_size)
            torch.save(G, os.path.join(save_path, pt_fname[:-3]+'.pt'))
        except OSError:
            pbar.set_description('%s - Broken H5' % (pt_fname))
            print(pt_fname, 'Broken')


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt-path', type=str, default= '../pt_files/')
    parser.add_argument('--save-path', type=str, default= '../pt_files_knn/')
    parser.add_argument('--radius', type=int, default= 49)
    parser.add_argument('--min-size', type=int, default= 200)
    parser.add_argument('--ratio', type=float, default= 0.6)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_parse()
    pt_path = args.pt_path
    save_path = args.save_path
    radius = args.radius
    min_size = args.min_size
    ratio = args.ratio
    os.makedirs(save_path, exist_ok=True)
    createDir_pttoPyG(pt_path, save_path, radius, min_size, ratio)