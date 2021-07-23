import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ids2017_dataset as md17
import ids2018_dataset as md18
import nslkdd_dataset as nsl
import creditcard_dataset as cc
import satellite_dataset_folded as sd
import lympho_dataset_folded as lymds
import cardio_dataset_folded as carmd
import thyroid_dataset_folded as thymd
import mammo_dataset_folded as mammd
import vowels_dataset_folded as vomd
import yeast_dataset_folded as yemd
import satimage_dataset_folded as satmd
import shuttle_dataset_folded as shmd
import seismic_dataset_folded as seismd
import musk_dataset_folded as muskmd
import annthyroid_dataset_folded as annmd

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from collections import Counter
import cudf, cuml
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cuml.cluster import KMeans as cuKMeans
from siamese_network_model import SiameseNetwork
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE

DATA_PATH_2017 = "/home/nivgold/datasets/IDS2017"
DATA_PATH_2018 = "/home/nivgold/datasets/IDS2018/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_PATH_NSLKDD = "/home/nivgold/datasets/NSL-KDD"
DATA_PATH_CREDITCARD = '/home/nivgold/datasets/Credit-Card-Fraud'
DATA_PATH_SATELLITE = '/home/nivgold/datasets/Satellite'
DATA_PATH_LYMPHO = '/home/nivgold/datasets/Lymphography'
DATA_PATH_CARDIO = '/home/nivgold/datasets/Cardiotocography'
DATA_PATH_THYROID = '/home/nivgold/datasets/Thyroid'
DATA_PATH_MAMMO = '/home/nivgold/datasets/Mammography'
DATA_PATH_VOWELS = '/home/nivgold/datasets/Vowels'
DATA_PATH_YEAST = '/home/nivgold/datasets/Yeast'
DATA_PATH_SATIMAGE = '/home/nivgold/datasets/Satimage'
DATA_PATH_SHUTTLE = '/home/nivgold/datasets/Shuttle'
DATA_PATH_SEISMIC = '/home/nivgold/datasets/Seismic'
DATA_PATH_MUSK = '/home/nivgold/datasets/Musk'
DATA_PATH_ANNTHYROID = '/home/nivgold/datasets/Annthyroid'

# ----------------------------------------------------------------------------------------------------------------------

# # Creating 2017 Pkls
# md17.IDS2017Dataset(DATA_PATH_2017, from_disk=False)

# ids17 = md17.IDS2017Dataset(DATA_PATH_2017, from_disk=True)

# print("X_pairs shape: ", ids17.X_pairs.shape)
# print("y_pairs shape: ", ids17.y_pairs.shape)

# print("X train shape: ", ids17.train_features.shape)
# print("Y train shape: ", ids17.train_labels.shape)

# print("X test shape: ", ids17.test_features.shape)
# print("Y test shape: ", ids17.test_labels.shape)

# print("X full: ", ids17.features_full.shape)
# print("Y full: ", ids17.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating 2018 Pkls
# md18.IDS2018Dataset(DATA_PATH_2018, from_disk=False)

# ids18 = md18.IDS2018Dataset(DATA_PATH_2018, from_disk=True)

# print("X_pairs shape: ", ids18.X_pairs.shape)
# print("y_pairs shape: ", ids18.y_pairs.shape)

# print("X train shape: ", ids18.train_features.shape)
# print("Y train shape: ", ids18.train_labels.shape)

# print("X test shape: ", ids18.test_features.shape)
# print("Y test shape: ", ids18.test_labels.shape)

# print("X full: ", ids18.features_full.shape)
# print("Y full: ", ids18.labels_full.shape)

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

# thyroid = thymd.ThyroidDataset(DATA_PATH_THYROID, from_disk=True)

# print("X_pairs shape: ", thyroid.X_pairs.shape)
# print("y_pairs shape: ", thyroid.y_pairs.shape)

# print("X train shape: ", thyroid.train_features.shape)
# print("Y train shape: ", thyroid.train_labels.shape)

# print("X test shape: ", thyroid.test_features.shape)
# print("Y test shape: ", thyroid.test_labels.shape)

# print("X full shape: ", thyroid.features_full.shape)
# print("Y full shape", thyroid.labels_full.shape)

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

# # Creating Vowels Pkls
# vowels = vomd.VowelsDataset(DATA_PATH_VOWELS, from_disk=False)

# vowels = vomd.VowelsDataset(DATA_PATH_VOWELS, from_disk=True)

# print("X_pairs shape: ", vowels.X_pairs.shape)
# print("y_pairs shape: ", vowels.y_pairs.shape)

# print("X train shape: ", vowels.train_features.shape)
# print("Y train shape: ", vowels.train_labels.shape)

# print("X test shape: ", vowels.test_features.shape)
# print("Y test shape: ", vowels.test_labels.shape)

# print("X full shape: ", vowels.features_full.shape)
# print("Y full shape", vowels.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Yeast Pkls
# yeast = yemd.YeastDataset(DATA_PATH_YEAST, from_disk=False)

# yeast = yemd.YeastDataset(DATA_PATH_YEAST, from_disk=True)

# print("X_pairs shape: ", yeast.X_pairs.shape)
# print("y_pairs shape: ", yeast.y_pairs.shape)

# print("X train shape: ", yeast.train_features.shape)
# print("Y train shape: ", yeast.train_labels.shape)

# print("X test shape: ", yeast.test_features.shape)
# print("Y test shape: ", yeast.test_labels.shape)

# print("X full shape: ", yeast.features_full.shape)
# print("Y full shape", yeast.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Satimage Pkls
# satimage = satmd.SatimageDataset(DATA_PATH_SATIMAGE, from_disk=False)

# satimage = satmd.SatimageDataset(DATA_PATH_SATIMAGE, from_disk=True)

# print("X_pairs shape: ", satimage.X_pairs.shape)
# print("y_pairs shape: ", satimage.y_pairs.shape)

# print("X train shape: ", satimage.train_features.shape)
# print("Y train shape: ", satimage.train_labels.shape)

# print("X test shape: ", satimage.test_features.shape)
# print("Y test shape: ", satimage.test_labels.shape)

# print("X full shape: ", satimage.features_full.shape)
# print("Y full shape", satimage.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Shuttle Pkls
# shuttle = shmd.ShuttleDataset(DATA_PATH_SHUTTLE, from_disk=False)

# shuttle = shmd.ShuttleDataset(DATA_PATH_SHUTTLE, from_disk=True)

# print("X_pairs shape: ", shuttle.X_pairs.shape)
# print("y_pairs shape: ", shuttle.y_pairs.shape)

# print("X train shape: ", shuttle.train_features.shape)
# print("Y train shape: ", shuttle.train_labels.shape)

# print("X test shape: ", shuttle.test_features.shape)
# print("Y test shape: ", shuttle.test_labels.shape)

# print("X full shape: ", shuttle.features_full.shape)
# print("Y full shape", shuttle.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Seismic Pkls
# seismic = seismd.SeismicDataset(DATA_PATH_SEISMIC, from_disk=False)

# seismic = seismd.SeismicDataset(DATA_PATH_SEISMIC, from_disk=True)

# print("X_pairs shape: ", seismic.X_pairs.shape)
# print("y_pairs shape: ", seismic.y_pairs.shape)

# print("X train shape: ", seismic.train_features.shape)
# print("Y train shape: ", seismic.train_labels.shape)

# print("X test shape: ", seismic.test_features.shape)
# print("Y test shape: ", seismic.test_labels.shape)

# print("X full shape: ", seismic.features_full.shape)
# print("Y full shape", seismic.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Musk Pkls
# musk = muskmd.MuskDataset(DATA_PATH_MUSK, from_disk=False)

# musk = muskmd.MuskDataset(DATA_PATH_MUSK, from_disk=True)

# print("X_pairs shape: ", musk.X_pairs.shape)
# print("y_pairs shape: ", musk.y_pairs.shape)

# print("X train shape: ", musk.train_features.shape)
# print("Y train shape: ", musk.train_labels.shape)

# print("X test shape: ", musk.test_features.shape)
# print("Y test shape: ", musk.test_labels.shape)

# print("X full shape: ", musk.features_full.shape)
# print("Y full shape", musk.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------

# # Creating Annthyroid Pkls
# annthyroid = annmd.AnnthyroidDataset(DATA_PATH_ANNTHYROID, from_disk=False)

# annthyroid = annmd.AnnthyroidDataset(DATA_PATH_ANNTHYROID, from_disk=True)

# print("X_pairs shape: ", annthyroid.X_pairs.shape)
# print("y_pairs shape: ", annthyroid.y_pairs.shape)

# print("X train shape: ", annthyroid.train_features.shape)
# print("Y train shape: ", annthyroid.train_labels.shape)

# print("X test shape: ", annthyroid.test_features.shape)
# print("Y test shape: ", annthyroid.test_labels.shape)

# print("X full shape: ", annthyroid.features_full.shape)
# print("Y full shape", annthyroid.labels_full.shape)

# ----------------------------------------------------------------------------------------------------------------------