# LNPL-MIL
The implementation of LNPL-MIL: Learning from Noisy Pseudo Labels for Promoting Multiple Instance Learning in Whole Slide Image (ICCV 2023).

## Installation
Please refer to our previous paper [HVTSurv](https://github.com/szc19990412/HVTSurv) for instructions on how to set up the environment. 

## Feature Generation and Pseudo-labels Assignment
Please refer to the [CLAM](https://github.com/mahmoodlab/CLAM) to embed WSIs into features.

We refer to the code of [suncet](https://github.com/facebookresearch/suncet) to get a supervised weakly classifier with limited labeled data. Then this weak classifier is used to assign pseudo labels for unlabeled data. 

## Top-K key instances selection
The data is structured in the following format: [score, coords, features]. In this format, 'score' represents the probability of being positive. 'Coords' represent the x and y coordinates, while 'features' represent the features extracted by a pretrained ResNet18 model.

``` python
python knn_feature.py --pt-path '../pt_files/' --save-path '../pt_files_knn/' --radius 50 --min-size 200 --ratio 0.4
```
We offer two implementation versions. If we opt for the iterative approach to obtain the super patch, using non-overlapping super patches can yield better performance. However, this may result in significantly higher CPU computational costs.

## Transformer Aware of Instance Order and Distribution in MIL

```python
for((FOLD=0;FOLD<4;FOLD++));
do
    python train.py --stage='train'\
    --config='Camelyon/TODMIL_995.yaml'  --gpus=0 --fold=$FOLD
    python train.py --stage='test'\
    --config='Camelyon/TODMIL_995.yaml'  --gpus=0 --fold=$FOLD
done
python metrics.py --config='Camelyon/TODMIL_995.yaml'
```






