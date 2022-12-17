# building the VAE architecture

import numpy as np
import tensorflow as tf
from tensorflow import keras


def kl_anneal(step, type='logistic', k=0.001, x0=100):
    """Function to dynamically change the KL term weight during model training

    Args:
        step (int): Current step in training
        type (str, optional): KL weight schedule; either logistic or linear. Defaults to 'logistic'.
        k (float, optional): k in Logisitc function. Defaults to 0.001.
        x0 (int, optional): x0 in logitstic function. Defaults to 100.

    Returns:
        float: KL term weight between 0-1
    """
    # KL term annealing
    if type == 'normal':
        pass
    if type == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif type == 'linear':
        return float(min(1, step/x0))


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector in latent space

    Args:
        keras (_type_): 

    Returns:
        _type_: latent tensor
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Kl_Annealing_Loss(tf.keras.layers.Layer):
    def __init__(self, beta, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var, z = inputs
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss_batch = -0.5*tf.reduce_mean(kl_loss)
        out = tf.math.multiply(self.beta, kl_loss_batch)
        return out


class Recon_Loss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, data, reconstruction):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(
                    data, reconstruction), -1
            ))
        return reconstruction_loss


def get_encoder(latent_dim=50):
    """The encoder neural network 

    Args:
        latent_dim (int, optional): Number of latent dimensions. Defaults to 50.

    Returns:
        Keras functional model: Encdoer nn 
    """
    encoder_inputs = keras.layers.Input(shape=(48, 22))
    h = encoder_inputs
    h = keras.layers.Conv1D(22, 2, strides=2, activation='relu')(h)
    h = keras.layers.Conv1D(44, 2, strides=2, activation='relu')(h)
    h = keras.layers.Conv1D(88, 2, strides=2, activation='relu')(h)
    h = keras.layers.Conv1D(176, 2, strides=2, activation='relu')(h)
    h = keras.layers.Flatten()(h)
    z_mean = keras.layers.Dense(
        latent_dim, kernel_initializer='random_normal')(h)
    z_log_var = keras.layers.Dense(
        latent_dim, kernel_initializer='random_normal')(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.models.Model(
        encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    # TODO encoder.load_weights("VAE_v9_pre_encoder_weights")
    return encoder


def get_decoder(latent_dim=50):
    """The decoder neural netweork

    Args:
        latent_dim (int, optional): Number of latent dimensions. Defaults to 50.

    Returns:
        Keras functional model: Decoder nn
    """
    latent_inputs = keras.layers.Input(shape=(latent_dim))
    h = latent_inputs
    h = keras.layers.Dense(3*176, kernel_initializer='random_normal')(h)
    h = keras.layers.Reshape((3, 176))(h)
    h = keras.layers.Conv1DTranspose(88, 2, strides=2)(h)
    h = keras.layers.Conv1DTranspose(44, 2, strides=2)(h)
    h = keras.layers.Conv1DTranspose(22, 2, strides=2)(h)
    h = keras.layers.Conv1DTranspose(22, 1, strides=2)(h)
    rnn_out = keras.layers.GRU(512, return_sequences=True)(h)
    processed_x = keras.layers.Conv1D(
        22, 1, activation=None, use_bias=True)(rnn_out)
    decoder_output = keras.layers.Softmax(axis=-1)(processed_x)
    decoder = keras.models.Model(latent_inputs, decoder_output, name='decoder')
    # TODO decoder.load_weights("VAE_v9_pre_decoder_weights")
    return decoder


class VAE(keras.Model):
    """The class describing the VAE loss function and training step 

    Args:
        keras (_type_): 
    """

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta_weight = 1.0

        self.beta_var = tf.Variable(
            self.beta_weight, trainable=False, name="Beta_annealing", validate_shape=False)

        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(
            name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            latent = self.encoder(data)
            reconstruction = self.decoder(latent[2])
            kl_loss = Kl_Annealing_Loss(self.beta_var)(latent)
            reconstruction_loss = Recon_Loss()(data, reconstruction)
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": self.beta_var
        }

    def call(self, data):
        return (self.decoder(self.encoder(data)[2]))
