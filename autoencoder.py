from tensorflow.keras import layers
from tensorflow.keras.models import Model
from typing import Tuple

class Autoencoder():
    def build(height:int, width:int, depth:int, nodes: Tuple[int], window: int, stride: int, latent_dim: int) -> Model:
        shape = (width, height, depth)
        inputs = layers.Input(shape)
        x = inputs
        for node in nodes:
            x = layers.Conv2D(node, (window,window), activation='relu', padding='same', strides=stride)(x)
        conv_shape = x.shape
        x = layers.Flatten()(x)
        x = layers.Dense(latent_dim + latent_dim)(x)
        latent = x
        encoder = Model(inputs, x, name = 'encoder')

        x = layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3])(x)
        x = layers.Reshape(target_shape = conv_shape[1:])(x)
        reverse_nodes = nodes[::-1]
        for node in reverse_nodes:
            x = layers.Conv2DTranspose(node, kernel_size=window, strides=stride, activation="relu", padding="same")(x)
        x = layers.Conv2D(depth, kernel_size = (window, window), activation="sigmoid", padding="same")(x)
        outputs = x
        decoder = Model(latent, outputs, name = 'decoder')
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        return autoencoder
