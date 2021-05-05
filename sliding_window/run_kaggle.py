import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import kaggle_dataset as kagmd
from solver import Solver
import tensorflow as tf
import time



gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

HOME_PATH = '/home/nivgold'
EPOCHS = 300
KAGGLE_DIM = 33

# testing ids2017 dataset
start_time = time.time()
kaggle_train_ds, kaggle_test_ds = kagmd.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("--Kaggle train_ds, test_ds ready after: ", end='')
get_execute_time(start_time, end_train_test)
solver_obj = Solver(kaggle_train_ds, kaggle_test_ds, epochs=EPOCHS, features_dim=KAGGLE_DIM)
encoder_path = '/home/nivgold/models/sliding_window_models/epochs_300_KAGGLE_encoder_weights.npy'
decoder_path = '/home/nivgold/models/sliding_window_models/epochs_300_KAGGLE_decoder_weights.npy'

# # TRAINING
# start_time = time.time()
# print("Start training...")
# solver_obj.train()
# end_training = time.time()
# print("---training finished after: ", end='')
# get_execute_time(start_time, end_training)
# # saving the trained weights
# solver_obj.save_weights(path='/home/nivgold/models/sliding_window_models', dataset_name='KAGGLE')

solver_obj.load_weights(encoder_path, decoder_path)

# TEST WITHOUT TTA
start_time = time.time()
print("Start testing...")
accuracy, precision, recall, f_score, auc = solver_obj.test()
end_testing = time.time()
print("---testing finished after: ", end='')
get_execute_time(start_time, end_testing)

# TEST WITH TTA

start_time = time.time()
print(f"Start testing with TTA...")
accuracy, precision, recall, f_score, auc = solver_obj.test_tta() 
end_tta_testing = time.time()
print("---TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)