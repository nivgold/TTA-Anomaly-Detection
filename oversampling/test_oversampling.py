import ids2017_dataset as md17
import ids2018_dataset as md18
import numpy as np

from imblearn.over_sampling import SMOTE
from collections import Counter

DATA_PATH_2017 = "/home/nivgold/TrafficLabelling"
DATA_PATH_2018 = "/home/nivgold/IT/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

# # Creating 2017 Pkls
# md17.IDS2017Dataset(DATA_PATH_2017, from_disk=False)

# ids17 = md17.IDS2017Dataset(DATA_PATH_2017, from_disk=True)
# print(ids17.train_features.shape)
# print(ids17.train_labels.shape)

# # Creaing 2018 Pkls
# md18.IDS2018Dataset(DATA_PATH_2018, from_disk=False)

ids18 = md18.IDS2018Dataset(DATA_PATH_2018, from_disk=True)
X_train = ids18.train_features
y_train = ids18.train_labels

# print("X_train shape: ", X_train.shape)
# print("y_train shape: ", y_train.shape)
#
# print("classes:", y_train.unique())

from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5)
print("start fit Nearest Neighbors...")
nn.fit(X_train[2:10].values)
print("Nearest Neighbors fitted.")

ind = nn.kneighbors(X_train[0:2].values, return_distance=False)
ind = np.squeeze(ind)
tta_batch_features = X_train.values[ind]
tta_batch_labels = y_train.values[ind]

# determine the number of augmentation
num_augmentation = 5

for tta_features, tta_labels in list(zip(tta_batch_features, tta_batch_labels)):
    fake_tta_features = [tta_feature for tta_feature in tta_features] + [tta_features[-1] for i in range(num_augmentation)]
    fake_tta_labels = [1 for i in fake_tta_features]
    tta_features_full = np.concatenate([tta_features, fake_tta_features])
    tta_labels_full = np.concatenate([tta_labels, fake_tta_labels])

    print("Before:")
    print(tta_features)
    print(tta_labels)
    sm = SMOTE(sampling_strategy='minority', k_neighbors=1)
    X_res, y_res = sm.fit_resample(tta_features_full, tta_labels_full)
    print("\n\nAfter:")
    tta_sample = X_res[-5:]
    print(tta_sample)
    print(y_res[-5:])

    break