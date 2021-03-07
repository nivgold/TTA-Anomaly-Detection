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

def save_model(solver_obj, datasets_path, models_path, dataset_name):
    # save a solver on disk
    train_ds = solver_obj.train_ds
    test_ds = solver_obj.test_ds

    # saving the tf.Datasets
    train_ds_path = datasets_path + "/train_ds"
    test_ds_path = datasets_path + "/test_ds"

    tf.data.experimental.save(train_ds, train_ds_path)
    tf.data.experimental.save(test_ds, test_ds_path)

    # saving the trained weights
    model_path = models_path + f"/epochs_{solver_obj.num_epochs}_{dataset_name}"

    solver_obj.encoder.save(model_path+"_encoder")
    solver_obj.decoder.save(model_path+"_decoder")

def load_solver(datasets_path, features_dim, model_path, num_epochs):
    # loading train_ds and test_ds from files
    train_ds_path = datasets_path + "/train_ds"
    test_ds_path = datasets_path + "/test_ds"
    train_ds = tf.data.experimental.load(train_ds_path,
                                          element_spec=(tf.TensorSpec(shape=(None, features_dim), dtype=tf.float64, name=None),
                                                        tf.TensorSpec(shape=(None,), dtype=tf.int16, name=None)))

    test_ds = tf.data.experimental.load(test_ds_path,
                                         element_spec=(tf.TensorSpec(shape=(None, features_dim), dtype=tf.float64, name=None),
                                                       tf.TensorSpec(shape=(None,), dtype=tf.int16, name=None)))

    encoder_path = model_path + "/encoder"
    encoder_model = tf.keras.models.load_model(encoder_path)
    decoder_path = model_path + "/decoder"
    decoder_model = tf.keras.models.load_model(decoder_path)

    solver_obj = Solver(train_ds, test_ds, num_epochs, features_dim)
    # updating the encoder and decoder models to be the trained ones
    solver_obj.encoder = encoder_model
    solver_obj.decoder = decoder_model

    return solver_obj


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