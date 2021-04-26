import ids2017_dataset as ids17

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
EPOCHS = 100
IDS17_DIM = 77

# testing ids2017 dataset
start_time = time.time()
ids17_train_ds, ids17_test_ds, ids17_train_full_ds, ids17_features_full = ids17.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("---IDS2017 train_ds, test_ds ready after: ", end='')
get_execute_time(start_time, end_train_test)
solver_obj = Solver(ids17_train_ds, ids17_test_ds, epochs=EPOCHS, features_dim=IDS17_DIM)
encoder_path = '/home/nivgold/models/epochs_100_IDS17_encoder_weights.npy'
decoder_path = '/home/nivgold/models/epochs_100_IDS17_decoder_weights.npy'

# # TRAINING
# start_time = time.time()
# print("Start training...")
# solver_obj.train()
# end_training = time.time()
# print("---training finished after: ", end='')
# get_execute_time(start_time, end_training)
# # saving the trained weights
# solver_obj.save_weights(path='/home/nivgold/models', dataset_name='IDS17')

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
accuracy, precision, recall, f_score, auc = solver_obj.test_tta(oversampling_method, num_neighbors=num_neighbors, num_augmentations=num_augmentations, knn_data=ids17_features_full)
end_tta_testing = time.time()
print("---TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)