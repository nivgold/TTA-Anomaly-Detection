# import ids2017_dataset as md17
# import ids2018_dataset as md18
# import numpy as np
#
# from imblearn.over_sampling import SMOTE
# from collections import Counter
#
# DATA_PATH_2017 = "/home/nivgold/TrafficLabelling"
# DATA_PATH_2018 = "/home/nivgold/IT/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
#
# # # Creating 2017 Pkls
# # md17.IDS2017Dataset(DATA_PATH_2017, from_disk=False)
#
# # ids17 = md17.IDS2017Dataset(DATA_PATH_2017, from_disk=True)
# # print(ids17.train_features.shape)
# # print(ids17.train_labels.shape)
#
# # # Creaing 2018 Pkls
# # md18.IDS2018Dataset(DATA_PATH_2018, from_disk=False)
#
# ids18 = md18.IDS2018Dataset(DATA_PATH_2018, from_disk=True)
# X_train = ids18.train_features
# y_train = ids18.train_labels
#
# # print("X_train shape: ", X_train.shape)
# # print("y_train shape: ", y_train.shape)
# #
# # print("classes:", y_train.unique())
#
# from sklearn.neighbors import NearestNeighbors
# import cuml, cudf
# from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
#
# # nn = NearestNeighbors(n_neighbors=5)
# print("Now with RAPIDS")
# print(X_train.shape)
# nn = cuNearestNeighbors(n_neighbors=5)
# print("start fit Nearest Neighbors...")
# X_rapids = cudf.DataFrame(X_train.values)
# nn.fit(X_rapids)
# print("Nearest Neighbors fitted.")
#
# ind = nn.kneighbors(X_train[0:2].values, return_distance=False)
# ind = np.squeeze(ind)
# tta_batch_features = X_train.values[ind]
# tta_batch_labels = y_train.values[ind]
#
# # determine the number of augmentation
# num_augmentation = 5
#
# for tta_features, tta_labels in list(zip(tta_batch_features, tta_batch_labels)):
#     fake_tta_features = [tta_feature for tta_feature in tta_features] + [tta_features[-1] for i in range(num_augmentation)]
#     fake_tta_labels = [1 for i in fake_tta_features]
#     tta_features_full = np.concatenate([tta_features, fake_tta_features])
#     tta_labels_full = np.concatenate([tta_labels, fake_tta_labels])
#
#     print("Before:")
#     print(tta_features)
#     # print(tta_labels)
#     sm = SMOTE(sampling_strategy='minority', k_neighbors=num_augmentation-1)
#     X_res, y_res = sm.fit_resample(tta_features_full, tta_labels_full)
#     print("\n\nAfter:")
#     tta_sample = X_res[-num_augmentation:]
#     print(tta_sample)
#     # print(y_res[-num_augmentation:])
#
#     break

import os
import numpy as np
from autoencdoer_model import Encoder
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

encoder_weights_path = "/home/nivgold/TTA-Anomaly-Detection/oversampling/out/models/epochs_100_IDS18_encoder_weights.npy"
encoder_weights = np.load(encoder_weights_path, allow_pickle=True)
decoder_weights_path = "/home/nivgold/TTA-Anomaly-Detection/oversampling/out/models/epochs_100_IDS18_decoder_weights.npy"
decoder_weights = np.load(decoder_weights_path, allow_pickle=True)
# print(os.listdir("/home/nivgold/TTA-Anomaly-Detection/oversampling/out/models"))

import ids2018_dataset as ids18
ids18_train_ds, ids18_test_ds = ids18.get_dataset("/home/nivgold", 32, from_disk=True)

encoder = Encoder(input_shape=76)
for batch_X_train, batch_y_train in ids18_train_ds:
    encoder(batch_X_train)
    break

# encoder.set_weights(encoder_weights)
# print(np.array(encoder.get_weights()))
# print()
# print()
# print()
# print(encoder_weights)

X = []
for batch_X_train, batch_y_train in ids18_train_ds.take(1000):
    X.append(batch_X_train)

X = np.array(X)
X = np.concatenate(X, axis=0)

import cudf, cuml
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

knn_model = cuNearestNeighbors(5)
knn_model.fit(X)

for batch_x_test, batch_y_test in ids18_test_ds:
    batch_x_test = np.array(batch_x_test)
    print(type(batch_x_test))
    print(batch_x_test.shape)
    neighbors_indices = knn_model.kneighbors(batch_x_test, return_distance=False)
    neighbors_indices = np.squeeze(neighbors_indices)

    print(neighbors_indices)
    break