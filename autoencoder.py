from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

# class Autoencoder:
#     def build(height, width, depth, filters = (32,64), latent_dim = 16):
#         input_shape =(width, height, depth)
#         channels_dimension = -1
#         inputs = layers.Input(shape=input_shape)
#         x = inputs
#         for filter in filters:
#             x = layers.Conv2D(filter, (3,3), strides = 2, padding="same")(x)
#             x = layers.LeakyReLU(alpha=0.2)(x)
#             x = layers.BatchNormalization(axis=channels_dimension)(x)

#         volumeSize = K.int_shape(x)
#         x = layers.Flatten()(x)
#         latent = layers.Dense(latent_dim)(x)
#         encoder = Model(inputs, latent, name = 'encoder')

#         latent_inputs = layers.Input(shape=(latent_dim,))
#         x = layers.Dense(np.prod(volumeSize[1:]))(latent_inputs)
#         x = layers.Reshape((volumeSize[1],volumeSize[2], volumeSize[3]))(x)

#         for filter in filters[::-1]:
#             x = layers.Conv2DTranspose(filter, (3,3), strides=2, padding='same')(x)
#             x = layers.LeakyReLU(alpha=0.2)(x)
#             x = layers.BatchNormalization(axis=channels_dimension)(x)
#         x = layers.Conv2D(depth, (3,3), padding="same")(x)
#         outputs = layers.Activation("sigmoid")(x)
#         decoder = Model(latent_inputs, outputs, name="decoder")
#         autoencoder = Model(inputs, decoder(encoder(inputs)), name = "autoencoder")
#         return encoder, decoder, autoencoder


class Autoencoder(Model):
    def __init__(self, height, width, depth, nodes, window, stride ):
        super(Autoencoder, self).__init__()
        self.shape = (width, height, depth)
        self.encoder = tf.keras.Sequential()
        for node in nodes:
            self.encoder.add(layers.Conv2D(node, (window,window), activation='relu', padding='same', strides=stride))
        self.decoder = tf.keras.Sequential()
        for node in nodes[::-1]:
            self.decoder.add(layers.Conv2DTranspose(node, kernel_size=window, strides=stride, activation="relu", padding="same"))
        self.decoder.add(layers.Conv2D(depth, kernel_size = (window, window), activation="sigmoid", padding="same"))
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
