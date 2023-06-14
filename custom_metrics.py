import keras.backend as K
import tensorflow as tf


def cc_met(obs_rate, est_rate):
    """Pearson correlation coefficient"""
    ax = -1
    x_mu = obs_rate - K.mean(obs_rate, axis=ax, keepdims=True)
    x_std = K.std(obs_rate, axis=ax, keepdims=True)
    y_mu = est_rate - K.mean(est_rate, axis=ax, keepdims=True)
    y_std = K.std(est_rate, axis=ax, keepdims=True)

    cc = K.mean(x_mu * y_mu, axis=ax, keepdims=True) / (x_std * y_std)
    cc = tf.where(tf.math.is_nan(cc), tf.ones_like(cc) * 0, cc);
    return  cc

def mean_squared_error(obs_rate, est_rate):
    """Mean squared error across samples"""
    return K.mean(K.square(est_rate - obs_rate), axis=-1, keepdims=True)


def rmse_met(obs_rate, est_rate):
    """Root mean squared error"""
    return K.sqrt(mean_squared_error(obs_rate, est_rate))
def norm_rmse_met(obs_rate, est_rate):
    """Root mean squared error"""
    rmse = K.sqrt(mean_squared_error(obs_rate, est_rate))
    average_obs = K.mean(obs_rate)
    return rmse/average_obs

def fev_met(obs_rate, est_rate):

    fvu = mean_squared_error(obs_rate, est_rate) / K.var(obs_rate, axis=-1, keepdims=True)
    fvu = tf.where(not tf.math.is_finite(fvu), tf.ones_like(fvu) * 1, fvu)
    return 1 - fvu
