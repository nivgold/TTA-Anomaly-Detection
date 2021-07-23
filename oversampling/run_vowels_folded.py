import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import vowels_dataset_folded as vomd

from solver_folded import Solver
import tensorflow as tf
tf.random.set_seed(1234)
import numpy as np
np.random.seed(1234)
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--neighbors", dest="num_neighbors", default=10, type=int, help="The number of neighbors to retrieve from the Neareset Neighbors model")
parser.add_argument("-a", "--augmentations", dest="num_augmentations", default=4, type=int, help="The number test-time augmentations to apply on every test sample")
parser.add_argument("-f", "--folds", dest="n_folds", default=10, type=int, help="The number of folds in the K-Fold cross-validation")
args = parser.parse_args()

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

HOME_PATH = '/home/nivgold'
EPOCHS = 500
VOWELS_DIM = 12

start_time = time.time()
vowels_X, vowels_y, vowels_pairs = vomd.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("--- Vowels dataset ready after: ", end='')
get_execute_time(start_time, end_train_test)
dataset_name = 'vowels'
weights_path = '/home/nivgold/models/oversampling_models'
solver_obj = Solver(X=vowels_X, y=vowels_y, n_folds=args.n_folds, dataset_name=dataset_name, epochs=EPOCHS, features_dim=VOWELS_DIM, siamese_data=vowels_pairs)

# TRAINING
start_time = time.time()
print("Start training...")
solver_obj.train_folded()
end_training = time.time()
print("---training finished after: ", end='')
get_execute_time(start_time, end_training)
# # saving the trained weights
# solver_obj.save_weights_folded(path=weights_path)

# solver_obj.load_weights_folded(weights_path)

# TEST WITHOUT TTA
start_time = time.time()
print("--- Start baseline testing...")
solver_obj.test_folded()
end_testing = time.time()
print("--- Baseline testing finished after: ", end='')
get_execute_time(start_time, end_testing)

# TEST WITH TTA
num_neighbors = args.num_neighbors
num_augmentations = args.num_augmentations

start_time = time.time()
print(f"--- Start TTA testing with: \t {num_neighbors} neighbors, {num_augmentations} TTA augmentations")
solver_obj.test_tta_folded(num_neighbors=num_neighbors, num_augmentations=num_augmentations)
end_tta_testing = time.time()
print("--- TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)

solver_obj.print_test_results()