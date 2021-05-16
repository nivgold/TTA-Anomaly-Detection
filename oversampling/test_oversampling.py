import ids2017_dataset as md17
import ids2018_dataset as md18
import nslkdd_dataset as nsl
import creditcard_dataset as cc
import satellite_dataset as sd
import lympho_dataset as lymds
import cardio_dataset as carmd
import thyroid_dataset as thymd
import mammo_dataset as mammd
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from collections import Counter
import cudf, cuml
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cuml.cluster import KMeans

DATA_PATH_2017 = "/home/nivgold/datasets/IDS2017"
DATA_PATH_2018 = "/home/nivgold/datasets/IDS2018/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_PATH_NSLKDD = "/home/nivgold/datasets/NSL-KDD"
DATA_PATH_CREDITCARD = '/home/nivgold/datasets/Credit-Card-Fraud'
DATA_PATH_SATELLITE = '/home/nivgold/datasets/Satellite'
DATA_PATH_LYMPHO = '/home/nivgold/datasets/Lymphography'
DATA_PATH_CARDIO = '/home/nivgold/datasets/Cardiotocography'
DATA_PATH_THYROID = '/home/nivgold/datasets/Thyroid'
DATA_PATH_MAMMO = '/home/nivgold/datasets/Mammography'

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
 
# print("X_pairs shape: ", nslds.X_pairs.shape)
# print("y_pairs shape: ", nslds.y_pairs.shape)

# print("X train shape: ", nslds.train_features.shape)
# print("Y train shape: ", nslds.train_labels.shape)

# print("X test shape: ", nslds.test_features.shape)
# print("Y test shape: ", nslds.test_labels.shape)

# print("X train full: ", nslds.train_features_full.shape)
# print("Y train full: ", nslds.train_labels_full.shape)

# print("X full: ", nslds.features_full.shape)
# print("Y full: ", nslds.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Credit Card Pkls
# cc.CreditCardDataset(DATA_PATH_CREDITCARD, from_disk=False)

# creditcard = cc.CreditCardDataset(DATA_PATH_CREDITCARD, from_disk=True)

# print("X_pairs shape: ", creditcard.X_pairs.shape)
# print("y_pairs shape: ", creditcard.y_pairs.shape)

# print("X train shape: ", creditcard.train_features.shape)
# print("Y train shape: ", creditcard.train_labels.shape)

# print("X full shape: ", creditcard.features_full.shape)
# print("Y full shape: ", creditcard.labels_full.shape)
# ----------------------------------------------------------------------------------------------------------------------

# # Creating Satellite Pkls
# satellite = sd.SatelliteDataset(DATA_PATH_SATELLITE, from_disk=False)

# satellite = sd.SatelliteDataset(DATA_PATH_SATELLITE, from_disk=True)

# print("X pairs shape: ", satellite.X_pairs.shape)
# print("y_pairs shape: ", satellite.y_pairs.shape)

# print("X train shape: ", satellite.train_features.shape)
# print("Y train shape: ", satellite.train_labels.shape)

# print("X full shape: ", satellite.features_full.shape)
# print("Y full shape: ", satellite.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Lympo Pkls
# lympho = lymds.LymphoDataset(DATA_PATH_LYMPHO, from_disk=False)

# lympho = lymds.LymphoDataset(DATA_PATH_LYMPHO, from_disk=True)

# print("X_pairs shape: ", lympho.X_pairs.shape)
# print("y_pairs shape: ", lympho.y_pairs.shape)

# print("X train shape: ", lympho.train_features.shape)
# print("Y train shape: ", lympho.train_labels.shape)

# print("X test shape: ", lympho.test_features.shape)
# print("Y test shape: ", lympho.test_labels.shape)

# print("X full shape: ", lympho.features_full.shape)
# print("Y full shape", lympho.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Cardio Pkls
# cardio = carmd.CardioDataset(DATA_PATH_CARDIO, from_disk=False)

# cardio = carmd.CardioDataset(DATA_PATH_CARDIO, from_disk=True)

# print("X_pairs shape: ", cardio.X_pairs.shape)
# print("y_pairs shape: ", cardio.y_pairs.shape)

# print("X train shape: ", cardio.train_features.shape)
# print("Y train shape: ", cardio.train_labels.shape)

# print("X test shape: ", cardio.test_features.shape)
# print("Y test shape: ", cardio.test_labels.shape)

# print("X full shape: ", cardio.features_full.shape)
# print("Y full shape", cardio.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Thyroid Pkls
# thyroid = thymd.ThyroidDataset(DATA_PATH_THYROID, from_disk=False)

thyroid = thymd.ThyroidDataset(DATA_PATH_THYROID, from_disk=True)

print("X_pairs shape: ", thyroid.X_pairs.shape)
print("y_pairs shape: ", thyroid.y_pairs.shape)

print("X train shape: ", thyroid.train_features.shape)
print("Y train shape: ", thyroid.train_labels.shape)

print("X test shape: ", thyroid.test_features.shape)
print("Y test shape: ", thyroid.test_labels.shape)

print("X full shape: ", thyroid.features_full.shape)
print("Y full shape", thyroid.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Mammography Pkls
# mammo = mammd.MammoDataset(DATA_PATH_MAMMO, from_disk=False)

# mammo = mammd.MammoDataset(DATA_PATH_MAMMO, from_disk=True)

# print("X_pairs shape: ", mammo.X_pairs.shape)
# print("y_pairs shape: ", mammo.y_pairs.shape)

# print("X train shape: ", mammo.train_features.shape)
# print("Y train shape: ", mammo.train_labels.shape)

# print("X test shape: ", mammo.test_features.shape)
# print("Y test shape: ", mammo.test_labels.shape)

# print("X full shape: ", mammo.features_full.shape)
# print("Y full shape", mammo.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------