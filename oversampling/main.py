import ids2017_dataset as ids17
import ids2018_dataset as ids18
from solver import Solver
import tensorflow as tf
import time

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

HOME_PATH = '/home/nivgold'
EPOCHS = 2
IDS17_DIM = 77
IDS18_DIM = 76

# testing ids2018 dataset
start_time = time.time()
ids18_train_ds, ids18_test_ds = ids18.get_dataset(HOME_PATH, 32, from_disk=True)
end_train_test = time.time()
print("---IDS2018 train_ds, test_ds ready after: ", end='')
get_execute_time(start_time, end_train_test)

solver_obj = Solver(ids18_train_ds, ids18_test_ds, epochs=EPOCHS, features_dim=IDS18_DIM)

if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

# TRAINING
start_time = time.time()
print("Start training...")
solver_obj.train()
end_training = time.time()
print("---training finished after: ", end='')
get_execute_time(start_time, end_training)

# # TEST WTHOUT TTA
# start_time = time.time()
# print("Start testing...")
# accuracy, precision, recall, f_score, auc = solver_obj.test()
# end_testing = time.time()
# print("---training finished after: ", end='')
# get_execute_time(start_time, end_testing)

# # TEST WITH TTA
# start_time = time.time()
# print("Start testing with TTA...")
# accuracy, precision, recall, f_score, auc = solver_obj.test_tta()
# end_tta_testing = time.time()
# print("---training finished after: ")
# get_execute_time(start_time, end_tta_testing)