import mammo_dataset as mammd

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
EPOCHS = 500
MAMMO_DIM = 6

start_time = time.time()
mammo_train_ds, mammo_test_ds, mammo_features_full = mammd.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("---Mammography train_ds, test_ds ready after: ", end='')
get_execute_time(start_time, end_train_test)
solver_obj = Solver(mammo_train_ds, mammo_test_ds, epochs=EPOCHS, features_dim=MAMMO_DIM)
dataset_name = 'mammo'
encoder_path = f'/home/nivgold/models/epochs_{EPOCHS}_{dataset_name}_encoder_weights.npy'
decoder_path = f'/home/nivgold/models/epochs_{EPOCHS}_{dataset_name}_decoder_weights.npy'

# # TRAINING
# start_time = time.time()
# print("Start training...")
# solver_obj.train()
# end_training = time.time()
# print("---training finished after: ", end='')
# get_execute_time(start_time, end_training)
# # saving the trained weights
# solver_obj.save_weights(path='/home/nivgold/models', dataset_name=dataset_name)

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
accuracy, precision, recall, f_score, auc = solver_obj.test_tta(oversampling_method, num_neighbors=num_neighbors, num_augmentations=num_augmentations, knn_data=mammo_features_full)
end_tta_testing = time.time()
print("---TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)