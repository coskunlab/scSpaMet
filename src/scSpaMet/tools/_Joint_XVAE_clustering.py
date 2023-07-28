import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda, Dropout
from keras import backend as K
import scanpy as sc
import pandas as pd

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

class ClusteringLayer(layers.Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')#the first parameter is shape and not name
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q 

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ClusteringLayerGaussian(ClusteringLayer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super().__init__(n_clusters,weights,alpha,**kwargs)
    
    def call(self,inputs,**kwargs):
        sigma=1.0
        q=K.sum(K.exp(-K.square(K.expand_dims(inputs,axis=1)-self.clusters)/(2.0*sigma*sigma)),axis=2)
        q=K.transpose(K.transpose(q)/K.sum(q,axis=1))
        return q

def Joint_XVAE_clustering(train_x_m1, train_x_m2, latent_dim=8, epochs=50, activa = 'relu', l2_penalty = 1e-5, netwidths_1=[16,8,8], netwidths_2=[64,32,16,8], n_neighbors=15, resolution=0.5, tol=0.005, iteration=5):
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

    features, _, encoded_data = vae.encoder.predict([train_x_m1, train_x_m2])
    x_reconstructed_m1, x_reconstructed_m2 = vae.decoder.predict(encoded_data)

    # Deep embedding
    adata0=sc.AnnData(features)
    sc.pp.neighbors(adata0, n_neighbors=n_neighbors,use_rep="X", random_state=0)
    sc.tl.leiden(adata0,resolution=resolution, random_state=0)
    Y_pred_init=adata0.obs['leiden']
    init_pred=np.asarray(Y_pred_init,dtype=int)
    if np.unique(init_pred).shape[0]<=1:
        #avoid only a cluster
        #print(np.unique(self.init_pred))
        exit("Error: There is only a cluster detected. The resolution:"+str(self.resolution)+"is too small, please choose a larger resolution!!")
    features=pd.DataFrame(adata0.X,index=np.arange(0,adata0.shape[0]))
    Group=pd.Series(init_pred,index=np.arange(0,adata0.shape[0]),name="Group")
    Mergefeature=pd.concat([features,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
    n_clusters=cluster_centers.shape[0]
    init_centroid=[cluster_centers]
    clustering_layer = ClusteringLayer(n_clusters,weights=init_centroid,name='clustering')(vae.encoder.output[0])
    desc = keras.Model(inputs=vae.encoder.input, outputs=clustering_layer)
    desc.compile(optimizer=keras.optimizers.SGD(0.01,0.9),loss='kld')

    y_pred_last = np.copy(init_pred)
    for ite in range(int(iteration)):
        q = desc.predict([train_x_m1, train_x_m2], verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        y_pred = q.argmax(1)
        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stop training.')
            break
        print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",tol)
        #train on whole dataset on prespecified batch_size
        desc.fit(x=[train_x_m1, train_x_m2],y=p,epochs=epochs,shuffle=True,verbose=True)
    
    y0=pd.Series(y_pred,dtype='category')
    # y0.cat.categories=range(0,len(y0.cat.categories))
    print("The final prediction cluster is:")
    x=y0.value_counts()
    print(x.sort_index(ascending=True))
    Embedded_z=vae.encoder.predict([train_x_m1, train_x_m2])
    return Embedded_z, q, [x_reconstructed_m1, x_reconstructed_m2]

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# class XVAE(object):
#     pass
