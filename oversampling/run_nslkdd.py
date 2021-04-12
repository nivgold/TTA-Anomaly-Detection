import nslkdd_dataset as nslmd

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
NSLKDD_DIM = 38

# testing ids2017 dataset
start_time = time.time()
nsl_train_ds, nsl_test_ds, nsl_train_full_ds = nslmd.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("--NSLKDD train_ds, test_ds ready after: ", end='')
get_execute_time(start_time, end_train_test)
solver_obj = Solver(nsl_train_ds, nsl_test_ds, epochs=EPOCHS, features_dim=NSLKDD_DIM)
encoder_path = '/home/nivgold/models/epochs_300_NSLKDD_encoder_weights.npy'
decoder_path = '/home/nivgold/models/epochs_300_NSLKDD_decoder_weights.npy'

# print("NSL train size: ", tf.data.experimental.cardinality(nsl_train_ds).numpy()*32)
# print("NSL train_full_ds size: ", tf.data.experimental.cardinality(nsl_train_full_ds).numpy()*32)

# # TRAINING
# start_time = time.time()
# print("Start training...")
# solver_obj.train()
# end_training = time.time()
# print("---training finished after: ", end='')
# get_execute_time(start_time, end_training)
# # saving the trained weights
# solver_obj.save_weights(path='/home/nivgold/models', dataset_name='NSLKDD')

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
accuracy, precision, recall, f_score, auc = solver_obj.test_tta(oversampling_method, num_neighbors=num_neighbors, num_augmentations=num_augmentations, knn_data=nsl_train_full_ds)
end_tta_testing = time.time()
print("---TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)