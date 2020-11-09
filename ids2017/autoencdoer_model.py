import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

tf.keras.backend.set_floatx('float64')


class Encoder(layers.Layer):
    def __init__(self, input_shape=231, latent_dim=16, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense1 = Dense(intermediate_dim, input_shape=(input_shape,), activation="relu")
        self.dense2 = Dense(latent_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        z = self.dense2(x)
        return z


class Decoder(layers.Layer):
    def __init__(self, input_shape=16, original_dim=231, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense1 = Dense(intermediate_dim, input_shape=(input_shape,), activation="relu")
        self.dense2 = Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


class SimpleAE(tf.keras.Model):
    def __init__(self, original_dim=115, intermediate_dim=64, latent_dim=32, name="autoencoder", **kwargs):
        super(SimpleAE, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim)

    def call(self, inputs):
        print("In Model's call")
        latent_space = self.encoder(inputs)
        return self.decoder(latent_space)
