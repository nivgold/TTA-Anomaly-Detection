import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nslkdd_dataset as nslmd

from solver import Solver
import tensorflow as tf
tf.random.set_seed(1234)
import numpy as np
np.random.seed(1234)
import time

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

HOME_PATH = '/home/nivgold'
EPOCHS = 300
NSLKDD_DIM = 38

start_time = time.time()
nsl_train_ds, nsl_test_ds, nsl_train_full_ds, nsl_features_full_ds, nsl_pairs = nslmd.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("--- NSLKDD dataset ready after: ", end='')
get_execute_time(start_time, end_train_test)
dataset_name = 'NSLKDD'
solver_obj = Solver(nsl_train_ds, nsl_test_ds, dataset_name=dataset_name, epochs=EPOCHS, features_dim=NSLKDD_DIM, knn_data=nsl_features_full_ds, siamese_data=nsl_pairs)
encoder_path = f'/home/nivgold/models/oversampling_models/epochs_{EPOCHS}_{dataset_name}_encoder_weights.npy'
decoder_path = f'/home/nivgold/models/oversampling_models/epochs_{EPOCHS}_{dataset_name}_decoder_weights.npy'

# # TRAINING
# start_time = time.time()
# print("Start training...")
# solver_obj.train()
# end_training = time.time()
# print("---training finished after: ", end='')
# get_execute_time(start_time, end_training)
# # saving the trained weights
# solver_obj.save_weights(path='/home/nivgold/models/oversampling_models', dataset_name='NSLKDD')

solver_obj.load_weights(encoder_path, decoder_path)

# TEST WITHOUT TTA
start_time = time.time()
print("--- Start baseline testing...")
solver_obj.test()
end_testing = time.time()
print("--- Baseline testing finished after: ", end='')
get_execute_time(start_time, end_testing)

# TEST WITH TTA
num_neighbors = 50
num_augmentations = 25

start_time = time.time()
print(f"--- Start TTA testing with: \t {num_neighbors} neighbors, {num_augmentations} TTA augmentations")
solver_obj.test_tta(num_neighbors=num_neighbors, num_augmentations=num_augmentations)
end_tta_testing = time.time()
print("--- TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)

solver_obj.print_test_results()
