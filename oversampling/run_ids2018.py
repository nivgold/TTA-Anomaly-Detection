import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ids2018_dataset as ids18

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
EPOCHS = 100
IDS18_DIM = 76

# testing ids2018 dataset
start_time = time.time()
ids18_train_ds, ids18_test_ds, ids18_features_full, ids18_pairs = ids18.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("---IDS2018 train_ds, test_ds ready after: ", end='')
get_execute_time(start_time, end_train_test)
solver_obj = Solver(ids18_train_ds, ids18_test_ds, epochs=EPOCHS, features_dim=IDS18_DIM, knn_data=ids18_features_full, siamese_data=ids18_pairs)
encoder_path = '/home/nivgold/models/oversampling_models/epochs_100_IDS18_encoder_weights.npy'
decoder_path = '/home/nivgold/models/oversampling_models/epochs_100_IDS18_decoder_weights.npy'

# # TRAINING
# start_time = time.time()
# print("Start training...")
# solver_obj.train()
# end_training = time.time()
# print("---training finished after: ", end='')
# get_execute_time(start_time, end_training)
# # saving the trained weights
# solver_obj.save_weights(path='/home/nivgold/models/oversampling_models', dataset_name='IDS18')

solver_obj.load_weights(encoder_path, decoder_path)

# TEST WITHOUT TTA
start_time = time.time()
print("Start testing...")
accuracy, precision, recall, f_score, auc = solver_obj.test()
end_testing = time.time()
print("---testing finished after: ", end='')
get_execute_time(start_time, end_testing)

# TEST WITH TTA
oversampling_method = "smote"
num_neighbors = 5
num_augmentations = 2

start_time = time.time()
print(f"Start testing with TTA... \t {oversampling_method}, {num_neighbors} neighbors, {num_augmentations} TTA augmentations")
accuracy, precision, recall, f_score, auc = solver_obj.test_tta(num_neighbors=num_neighbors, num_augmentations=num_augmentations)
end_tta_testing = time.time()
print("---TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)