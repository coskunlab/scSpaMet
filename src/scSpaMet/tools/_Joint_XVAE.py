import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda, Dropout
from keras import backend as K

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0.0, stddev=0.1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class XVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(XVAE, self).__init__(**kwargs)
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
        data_m1 = data[0][0]
        data_m2 = data[0][1]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstructions = self.decoder(z)
            # reconstruction_loss_m1 = tf.reduce_mean(
            #     keras.losses.binary_crossentropy(data_m1, reconstructions[0])
            # )
            # reconstruction_loss_m2 = tf.reduce_mean(
            #     keras.losses.binary_crossentropy(data_m2, reconstructions[1])
            # )
            reconstruction_loss_m1 = keras.losses.mean_squared_error(data_m1, reconstructions[0])
            reconstruction_loss_m2 = keras.losses.mean_squared_error(data_m2, reconstructions[1])
            reconstruction_loss = reconstruction_loss_m1 + 0.5*reconstruction_loss_m2
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss*0.01
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

def Joint_XVAE(train_x_m1, train_x_m2, latent_dim=8, epochs=50, activa = 'relu', l2_penalty = 1e-5, netwidths_1=[16,8,8], netwidths_2=[64,32,16,8]):
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    encoder_inputs_m1 = Input(shape=(train_x_m1.shape[1],))
    encoder_inputs_m2 = Input(shape=(train_x_m2.shape[1],))
    inputs = [encoder_inputs_m1, encoder_inputs_m2]

    kernel_init_func = tf.keras.initializers.GlorotUniform()

    # Encoder Modality 1
    m1_enc1 = Dense(netwidths_1[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(encoder_inputs_m1)
    m1_enc1 = BN()(m1_enc1)
    m1_enc2 = Dense(netwidths_1[1],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(m1_enc1)
    m1_enc2 = BN()(m1_enc2)
    # Encoder Modality 2
    m2_enc1 = Dense(netwidths_2[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(encoder_inputs_m2)
    m1_enc1 = BN()(m1_enc1)
    m2_enc2 = Dense(netwidths_2[1],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(m2_enc1)
    m2_enc2 = BN()(m2_enc2)
    m2_enc3 = Dense(netwidths_2[2],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(m2_enc1)
    m2_enc3 = BN()(m2_enc3)

    # Concatenate
    x_concat = Concatenate(axis=-1)([m1_enc2, m2_enc3])
    x_concat = Dense(latent_dim,activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(x_concat) 
    x_concat = BN()(x_concat)

    # Embedding layer
    z_mean = Dense(latent_dim, name="z_mean")(x_concat)
    z_log_var = Dense(latent_dim, name="z_log_var")(x_concat)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs , [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = Dense(latent_dim,activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(latent_inputs) 
    x = BN()(x)

    # Decoder m1
    m1_dec1 = Dense(netwidths_1[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(x)
    m1_dec1 = BN()(m1_dec1)
    m1_dec2 = Dense(train_x_m1.shape[1],activation="sigmoid",kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(m1_dec1)
    m1_dec2 = BN()(m1_dec2) 
    # Decoder m2
    m2_dec1 = Dense(netwidths_2[1],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(x)
    m2_dec1 = BN()(m2_dec1)
    m2_dec2 = Dense(netwidths_2[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(x)
    m2_dec2 = BN()(m2_dec2)
    m2_dec3 = Dense(train_x_m2.shape[1],activation="sigmoid",kernel_initializer=kernel_init_func,kernel_regularizer=regularizers.L2(l2_penalty))(m2_dec2) 
    m2_dec3 = BN()(m2_dec3)
    
    
    decoder = keras.Model(latent_inputs, [m1_dec2, m2_dec3], name="decoder")

    vae = XVAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit([train_x_m1, train_x_m2], shuffle=True, epochs=epochs, verbose=2)

    latent_embeddings, _, encoded_data = vae.encoder.predict([train_x_m1, train_x_m2])
    x_reconstructed_m1, x_reconstructed_m2 = vae.decoder.predict(encoded_data)
    return latent_embeddings, x_reconstructed_m1, x_reconstructed_m2

