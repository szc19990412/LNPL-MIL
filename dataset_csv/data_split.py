import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

test_dict = {'Tumor': 1, 'Normal': 0}
test_csv = pd.read_csv('CAMELYON16/testing/reference.csv', header=None)
test_csv.loc[:, 1] = test_csv.loc[:, 1].map(lambda x: test_dict[x])
test_name = test_csv.loc[:, 0].values.tolist()
test_label = test_csv.loc[:, 1].values.tolist()

train_val = pd.read_csv('../dataset_csv/camelyon16/fold0_.csv', index_col=0)
train_name = train_val.loc[:, 'train'].values.tolist()
train_label = train_val.loc[:, 'train_label'].values.tolist()
val_name = train_val.loc[:, 'val'].dropna().values.tolist()
val_label = train_val.loc[:, 'val_label'].dropna().values.tolist()

train_val_name = train_name + val_name
train_val_label = train_label + val_label

count = 0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in skf.split(train_val_name, train_val_label):
    train_name = [train_val_name[idx] for idx in train_index]
    train_label = [train_val_label[idx] for idx in train_index]
    val_name = [train_val_name[idx] for idx in val_index]
    val_label = [train_val_label[idx] for idx in val_index]

    df_train = pd.DataFrame({'train': train_name, 'train_label': train_label})
    df_val = pd.DataFrame({'val': val_name, 'val_label': val_label})
    df_test = pd.DataFrame({'test': test_name, 'test_label': test_label})
    df = pd.concat([df_train, df_val, df_test], axis=1)
    df.to_csv(f'../dataset_csv/camelyon16/fold{count}.csv')

    count = count + 1
    if count == 4:
        break