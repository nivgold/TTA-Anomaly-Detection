import ids2017_dataset as ids17
import ids2018_dataset as ids18
from solver import Solver
import tensorflow as tf
import time

# import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

HOME_PATH = '/home/nivgold'
EPOCHS = 1
IDS17_DIM = 77
IDS18_DIM = 76

# testing ids2018 dataset
start_time = time.time()
ids18_train_ds, ids18_test_ds = ids18.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("---IDS2018 train_ds, test_ds ready after: ", end='')
get_execute_time(start_time, end_train_test)

solver_obj = Solver(ids18_train_ds, ids18_test_ds, epochs=EPOCHS, features_dim=IDS18_DIM)

# TRAINING
start_time = time.time()
print("Start training...")
solver_obj.train()
end_training = time.time()
print("---training finished after: ", end='')
get_execute_time(start_time, end_training)
# saving the trained weights
solver_obj.save_weights(path='/home/nivgold/TTA-Anomaly-Detection/oversampling/out/models', dataset_name='IDS18')

encoder_path = '/home/nivgold/TTA-Anomaly-Detection/oversampling/out/models/epochs_100_IDS18_encoder_weights.npy'
decoder_path = '/home/nivgold/TTA-Anomaly-Detection/oversampling/out/models/epochs_100_IDS18_decoder_weights.npy'
solver_obj.load_weights(encoder_path, decoder_path)

# TEST WTHOUT TTA
start_time = time.time()
print("Start testing...")
accuracy, precision, recall, f_score, auc = solver_obj.test()
end_testing = time.time()
print("---testing finished after: ", end='')
get_execute_time(start_time, end_testing)

# TEST WITH TTA
start_time = time.time()
print("Start testing with TTA...")
accuracy, precision, recall, f_score, auc = solver_obj.test_tta("smote")
end_tta_testing = time.time()
print("---TTA testing finished after: ", end='')
get_execute_time(start_time, end_tta_testing)