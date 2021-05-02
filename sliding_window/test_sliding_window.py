import ids2017_dataset as md17
import ids2018_dataset as md18
import nslkdd_dataset as nsl

import numpy as np
import pandas as pd

DATA_PATH_2017 = "/home/nivgold/datasets/IDS2017"
DATA_PATH_2018 = "/home/nivgold/datasets/IDS2018/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_PATH_NSLKDD = "/home/nivgold/datasets/NSL-KDD"

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

# # Creating NSL-KDD Pkls
# nsl.NSLKDDDataset(DATA_PATH_NSLKDD, from_disk=False)

# nslds = nsl.NSLKDDDataset(DATA_PATH_NSLKDD, from_disk=True)
#
# print("X train shape: ", nslds.train_features.shape)
# print("Y train shape: ", nslds.train_labels.shape)
#
# print("X test shape: ", nslds.test_features.shape)
# print("Y test shape: ", nslds.test_labels.shape)
#
# print("X train full: ", nslds.train_features_full.shape)
# print("Y train full: ", nslds.train_labels_full.shape)
#
# print("X full: ", nslds.features_full.shape)
# print("Y full: ", nslds.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------