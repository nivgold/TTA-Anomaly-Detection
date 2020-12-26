import ids2017_dataset as ids17
import ids2018_dataset as ids18
from solver import Solver
import time

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

HOME_PATH = '/home/nivgold'
EPOCHS = 300

# testing ids2018 dataset
start_time = time.time()
ids18_train_ds, ids18_test_ds = ids18.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("---IDS2018 train_ds, test_ds ready after: ", end='')
get_execute_time(start_time, end_train_test)


# saving the tf.Dataset to a file
# import tensorflow as tf
# tf.data.experimental.save(train_ds, '/content/drive/My Drive/TTA Anomaly Detection/Simple Autoencoder/train_ds')
# tf.data.experimental.save(test_ds, '/content/drive/My Drive/TTA Anomaly Detection/Simple Autoencoder/test_ds')

# # loading train_ds and test_ds from files
# train_ds2 = tf.data.experimental.load('/content/drive/My Drive/TTA Anomaly Detection/Simple Autoencoder/train_ds',
#                                       element_spec=(tf.TensorSpec(shape=(None, 231), dtype=tf.float64, name=None),
#                                                     tf.TensorSpec(shape=(None,), dtype=tf.int16, name=None)))
#
# test_ds2 = tf.data.experimental.load('/content/drive/My Drive/TTA Anomaly Detection/Simple Autoencoder/test_ds',
#                                      element_spec=(tf.TensorSpec(shape=(None, 231), dtype=tf.float64, name=None),
#                                                    tf.TensorSpec(shape=(None,), dtype=tf.int16, name=None)))

solver = Solver(ids18_train_ds, ids18_test_ds, epochs=EPOCHS, features_dim=228)

# TRAINING
start_time = time.time()
print("Start training...")
solver.train()
end_training = time.time()
print("---training finished after: ", end='')
get_execute_time(start_time, end_training)

# TEST WTHOUT TTA
start_time = time.time()
print("Start testing...")
accuracy, precision, recall, f_score, auc = solver.test()
end_testing = time.time()
print("---training finished after: ", end='')
get_execute_time(start_time, end_testing)

# TEST WITH TTA
start_time = time.time()
print("Start testing with TTA...")
accuracy, precision, recall, f_score, auc = solver.test_tta()
end_tta_testing = time.time()
print("---training finished after: ")
get_execute_time(start_time, end_tta_testing)