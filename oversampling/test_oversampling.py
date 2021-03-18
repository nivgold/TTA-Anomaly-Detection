import ids2017_dataset as md17
import ids2018_dataset as md18
import numpy as np

from imblearn.over_sampling import SMOTE
from collections import Counter

DATA_PATH_2017 = "/home/nivgold/TrafficLabelling"
DATA_PATH_2018 = "/home/nivgold/IT/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

# Creating 2017 Pkls
md17.IDS2017Dataset(DATA_PATH_2017, from_disk=False)

# ids17 = md17.IDS2017Dataset(DATA_PATH_2017, from_disk=True)
# print(ids17.train_features.shape)
# print(ids17.train_labels.shape)

# # Creaing 2018 Pkls
# md18.IDS2018Dataset(DATA_PATH_2018, from_disk=False)

# ids18 = md18.IDS2018Dataset(DATA_PATH_2018, from_disk=True)
# X_train = ids18.train_features
# y_train = ids18.train_labels

# print("X_train shape: ", X_train.shape)
# print("y_train shape: ", y_train.shape)
#
# print("classes:", y_train.unique())