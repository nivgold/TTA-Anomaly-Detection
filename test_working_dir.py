import os
import pandas as pd
import numpy as np
import time
from tensorflow import keras
import my_dataset as md

def get_model(input_shape=231, latent_dim=16, intermediate_dim=64):
    inputs = keras.Input(shape=(input_shape, ))
    encoder_inter = keras.layers.Dense(intermediate_dim, activation="relu")(inputs)
    middle_layer = keras.layers.Dense(latent_dim, activation="relu")(encoder_inter)
    decoder_inter = keras.layers.Dense(intermediate_dim, activation="relu")(middle_layer)
    outputs = keras.layers.Dense(input_shape, activation="sigmoid")(decoder_inter)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# full_df = pd.read_pickle('/home/nivgold/pkls/full_df_from_my_dataset.pkl')
ids17 = md.IDS2017Dataset('/home/nivgold', None, from_disk=True)
print(len(ids17.tta_features))
print(len(ids17.test_features))
print(len(ids17.test_labels))