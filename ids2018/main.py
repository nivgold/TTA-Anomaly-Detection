from ids2017 import my_dataset as md
from ids2017.solver import Solver
from ids2017.test import get_execute_time
import time


HOME_PATH = '/home/nivgold'
EPOCHS = 300

# start_time = time.time()
# full_df = pd.read_pickle(f'{HOME_PATH}/pkls/full_df_from_my_dataset.pkl')
# end_full_df = time.time()
# print("---full_df ready after: ", end='')
# get_execute_time(start_time, end_full_df)


start_time = time.time()
train_ds, test_ds = md.get_dataset(HOME_PATH, 32, full_df=None, from_disk=True)
end_train_test = time.time()
print("---train_ds, test_ds ready after: ", end='')
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

solver = Solver(train_ds, test_ds, epochs=EPOCHS)

# # -----Solver train func -------
# class Encoder(tf.keras.layers.Layer):
#   def __init__(self, input_shape=231 ,latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
#     super(Encoder, self).__init__(name=name, **kwargs)
#     self.dense1 = tf.keras.layers.Dense(intermediate_dim, input_shape=(input_shape,), activation="relu")
#     self.dense2 = tf.keras.layers.Dense(latent_dim)

#   def call(self, inputs):
#     x = self.dense1(inputs)
#     z = self.dense2(x)
#     return z

# class Decoder(tf.keras.layers.Layer):
#   def __init__(self, input_shape=32 ,original_dim=231, intermediate_dim=64, name="decoder", **kwargs):
#     super(Decoder, self).__init__(name=name, **kwargs)
#     self.dense1 = tf.keras.layers.Dense(intermediate_dim, input_shape=(input_shape,) ,activation="relu")
#     self.dense2 = tf.keras.layers.Dense(original_dim, activation="sigmoid")
  
#   def call(self, inputs):
#     x = self.dense1(inputs)
#     return self.dense2(x)

# encoder = Encoder()
# decoder = Decoder()
# epochs = 32
# optimizer = tf.keras.optimizers.Adam()
# loss_func = tf.keras.losses.MeanSquaredError()
# for epoch in range(epochs):
#       epoch_loss_mean = tf.keras.metrics.Mean()
#       for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
#         with tf.GradientTape() as tape:
#           latent_var = encoder(x_batch_train)
          
#           outputs = decoder(latent_var)
#           loss = loss_func(x_batch_train, outputs)
          
#           trainable_vars = encoder.trainable_variables \
#                         + decoder.trainable_variables

#         grads = tape.gradient(loss, trainable_vars)
#         optimizer.apply_gradients(zip(grads, trainable_vars))
#         # keep track of the metrics
#         epoch_loss_mean.update_state(loss)

#         break
#       # update metrics after each epoch
#       print(f'Epoch {epoch} loss mean: {epoch_loss_mean.result()}')
#       break
# # -----Solver train func -------

start_time = time.time()
print("Start training...")
solver.train()
end_training = time.time()
print("---training finished after: ", end='')
get_execute_time(start_time, end_training)

start_time = time.time()
print("Start testing...")
accuracy, precision, recall, f_score, auc = solver.test()
end_testing = time.time()
print("---training finished after: ", end='')
get_execute_time(start_time, end_testing)

start_time = time.time()
print("Start testing with TTA...")
accuracy, precision, recall, f_score, auc = solver.test_tta()
end_tta_testing = time.time()
print("---training finished after: ")
get_execute_time(start_time, end_tta_testing)