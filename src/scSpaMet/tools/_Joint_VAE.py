import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import BatchNormalization as BN

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0.0, stddev=0.1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss*0.1
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def Joint_VAE(train_x, latent_dim=16, epochs=50, activa = 'relu',l2_penalty = 1e-5, netwidths=[128,64]):
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    encoder_inputs = keras.Input(shape=(train_x.shape[1],))
    kernel_init_func = tf.keras.initializers.GlorotNormal()

    # Encoder
    enc1 = layers.Dense(netwidths[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(encoder_inputs)
    enc1 = BN()(enc1)
    enc2 = layers.Dense(netwidths[1],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(enc1)
    enc2 = BN()(enc2)
    z_mean = layers.Dense(latent_dim, name="z_mean")(enc2)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(enc2)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    dec1 = layers.Dense(netwidths[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(latent_inputs) 
    dec1 = BN()(dec1)
    dec2 = layers.Dense(train_x.shape[1],activation="sigmoid",kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(dec1) 
    dec2 = BN()(dec2)
    decoder = keras.Model(latent_inputs, dec2, name="decoder")

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(train_x, shuffle=True, epochs=epochs, verbose=2)

    latent_embeddings, _, encoded_data = vae.encoder.predict(train_x)
    x_reconstructed = vae.decoder.predict(encoded_data)
    return latent_embeddings, x_reconstructed