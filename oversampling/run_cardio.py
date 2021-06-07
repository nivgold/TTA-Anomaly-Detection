import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cardio_dataset as carmd

from solver import Solver
import tensorflow as tf
tf.random.set_seed(1234)
import numpy as np
np.random.seed(1234)
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--neighbors", dest="num_neighbors", default=10, type=int, help="The number of neighbors to retrieve from the Neareset Neighbors model")
parser.add_argument("-a", "--augmentations", dest="num_augmentations", default=4, type=int, help="The number test-time augmentations to apply on every test sample")
args = parser.parse_args()

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

HOME_PATH = '/home/nivgold'
EPOCHS = 500
CARDIO_DIM = 21

start_time = time.time()
cardio_train_ds, cardio_test_ds, cardio_features_full, cardio_pairs = carmd.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("--- Cardio dataset ready after: ", end='')
get_execute_time(start_time, end_train_test)
dataset_name = 'cardio'
solver_obj = Solver(cardio_train_ds, cardio_test_ds, dataset_name=dataset_name, epochs=EPOCHS, features_dim=CARDIO_DIM, knn_data=cardio_features_full, siamese_data=cardio_pairs)
encoder_path = f'/home/nivgold/models/oversampling_models/epochs_{EPOCHS}_{dataset_name}_encoder_weights.npy'
decoder_path = f'/home/nivgold/models/oversampling_models/epochs_{EPOCHS}_{dataset_name}_decoder_weights.npy'

# TRAINING
start_time = time.time()
print("Start training...")
solver_obj.train()
end_training = time.time()
print("---training finished after: ", end='')
get_execute_time(start_time, end_training)
# # saving the trained weights
# solver_obj.save_weights(path='/home/nivgold/models/oversampling_models', dataset_name=dataset_name)

# solver_obj.load_weights(encoder_path, decoder_path)

# TEST WITHOUT TTA
start_time = time.time()
print("--- Start baseline testing...")
solver_obj.test()
end_testing = time.time()
print("--- Baseline testing finished after: ", end='')
get_execute_time(start_time, end_testing)

# TEST WITH TTA
num_neighbors = args.num_neighbors
num_augmentations = args.num_augmentations

start_time = time.time()
print(f"--- Start TTA testing with: \t {num_neighbors} neighbors, {num_augmentations} TTA augmentations")
solver_obj.test_tta(num_neighbors=num_neighbors, num_augmentations=num_augmentations)
end_tta_testing = time.time()
print("--- TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)

solver_obj.print_test_results()
