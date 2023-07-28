import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }


def Joint_AE(train_x, latent_dim=16, epochs=50, activa = 'relu',l2_penalty = 1e-5, netwidths=[128,64]):
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    encoder_inputs = keras.Input(shape=(train_x.shape[1],))
    kernel_init_func = tf.keras.initializers.GlorotNormal()

    # Encoder
    enc1 = layers.Dense(netwidths[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(encoder_inputs)
    enc2 = layers.Dense(netwidths[1],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(enc1)
    z = layers.Dense(latent_dim,activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(enc2)
    encoder = keras.Model(encoder_inputs, z, name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    dec3 = layers.Dense(netwidths[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(latent_inputs) 
    dec4 = layers.Dense(train_x.shape[1],activation="sigmoid",kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(dec3) 
    decoder = keras.Model(latent_inputs, dec4, name="decoder")

    vae = AE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(train_x, shuffle=True, epochs=epochs, verbose=2)

    latent_embeddings = vae.encoder.predict(train_x)
    x_reconstructed = vae.decoder.predict(latent_embeddings)
    return latent_embeddings, x_reconstructed