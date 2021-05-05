# import ids2017_dataset as md17
# import ids2018_dataset as md18
import nslkdd_dataset as nsl
import kaggle_dataset as kagmd

import numpy as np
import pandas as pd
import os

DATA_PATH_2017 = "/home/nivgold/datasets/IDS2017"
DATA_PATH_2018 = "/home/nivgold/datasets/IDS2018/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_PATH_NSLKDD = "/home/nivgold/datasets/NSL-KDD"
DATA_PATH_YAHOO = "/home/nivgold/datasets/Yahoo"
DATA_PATH_KAGGLE =  "/home/nivgold/datasets/Kaggle"

# ----------------------------------------------------------------------------------------------------------------------

# # Creating 2017 Pkls
# md17.IDS2017Dataset(DATA_PATH_2017, from_disk=False)

# ids17 = md17.IDS2017Dataset(DATA_PATH_2017, from_disk=True)
# print(ids17.train_features.shape)
# print(np.unique(ids17.train_labels))
# print(ids17.train_features_full.shape)
# print(np.unique(ids17.train_labels_full))
# print(ids17.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating 2018 Pkls
# md18.IDS2018Dataset(DATA_PATH_2018, from_disk=False)

# ids18 = md18.IDS2018Dataset(DATA_PATH_2018, from_disk=True)
# X_train = ids18.train_features
# full_features = ids18.features_full
# y_train = ids18.train_labels
#
# print("X_train shape: ", X_train.shape)
# print("full features shape: ", full_features.shape)
# print("y_train shape: ", y_train.shape)
#
# print("classes:", y_train.unique())

# ids18_train_ds, ids18_test_ds, ids18_features_full = md18.get_dataset(DATA_PATH_2018, 32, from_disk=True)

# ----------------------------------------------------------------------------------------------------------------------

# Creating NSL-KDD Pkls
# nsl.NSLKDDDataset(DATA_PATH_NSLKDD, from_disk=False)

# nslkdd = nsl.NSLKDDDataset(DATA_PATH_NSLKDD, from_disk=True)
# print("X TRAIN SHAPE: ", nslkdd.train_features.shape)
# print(nslkdd.train_features)
# print("Y TRAIN SHAPE: ", nslkdd.train_labels.shape)
# print("X TEST SHAPE: ", nslkdd.test_features.shape)
# print("Y TEST SHAPE: ", nslkdd.test_labels.shape)


# ----------------------------------------------------------------------------------------------------------------------

# Yahoo Dataset

# df_train = []
# df_test = []
# num_files = len(os.listdir(DATA_PATH_YAHOO))
# train_last_index = int(0.8*num_files)
# for file_index in range(1, num_files+1):
    # file_name = f"real_{file_index}.csv"
    # file_path = os.path.join(DATA_PATH_YAHOO, file_name)
    # if file_index <= train_last_index:
        # df_train.append(pd.read_csv(file_path))
    # else:
        # df_test.append(pd.read_csv(file_path))


# df_train = pd.concat(df_train, axis=0).reset_index(drop=True).drop(columns=['timestamp'], axis=1)
# df_test = pd.concat(df_test, axis=0).reset_index(drop=True).drop(columns=['timestamp'], axis=1)
# print(df_train)
# print("value counts: ", df_train['is_anomaly'].value_counts())
# print("anomaly locations: ", df_train[df_train['is_anomaly']==1])

# Creating Yahoo Pkls
# nsl.NSLKDDDataset(DATA_PATH_NSLKDD, from_disk=False)

# nslkdd = nsl.NSLKDDDataset(DATA_PATH_NSLKDD, from_disk=True)
# print("X TRAIN SHAPE: ", nslkdd.train_features.shape)
# print(nslkdd.train_features)
# print("Y TRAIN SHAPE: ", nslkdd.train_labels.shape)
# print("X TEST SHAPE: ", nslkdd.test_features.shape)
# print("Y TEST SHAPE: ", nslkdd.test_labels.shape)


# ----------------------------------------------------------------------------------------------------------------------

# Creating Kaggle Dataset pkls
# kagmd.KaggleDataset(DATA_PATH_KAGGLE, from_disk=False)

kaggle = kagmd.KaggleDataset(DATA_PATH_KAGGLE, from_disk=True)
print("X TRAIN SHAPE: ", kaggle.train_features.shape)
print("Y TRAIN SHAPE: ", kaggle.train_labels.shape)
print("X TEST SHAPE: ", kaggle.test_features.shape)
print("Y TEST SHAPE: ", kaggle.test_labels.shape)

print("test anomalies: ", kaggle.test_labels.value_counts())

# ----------------------------------------------------------------------------------------------------------------------
