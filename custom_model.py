from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1, l2
import tensorflow as tf
from tensorflow.keras.layers import Input
import numpy as np




def img_transformation_network(forward_mdl, inputs, n_out, model_args):
	if model_args['n_out'] == 32:
		strides = 4
	elif model_args['n_out'] == 16:
		strides = 8
	elif model_args['n_out'] == 8:
		strides = 16
	elif model_args['n_out'] == 4:
		strides = 32
	else:
		strides = 2


	if model_args['kernal_size'] <= 2:
		x = layers.Conv2D(model_args['n_channel'], 
						kernel_size = (model_args['kernal_size'],model_args['kernal_size']),
						strides=strides, 
						padding = 'valid',
						kernel_regularizer=l2(model_args['l2_reg']))(inputs)

	else:
		x = layers.Conv2D(filters = model_args['n_channel'], 
						kernel_size = (model_args['kernal_size'],model_args['kernal_size']),
						strides=strides, 
						padding = 'same',
						kernel_regularizer=l2(model_args['l2_reg']))(inputs)

	if model_args['n_channel'] > 1:

		x = layers.Conv2D(1, 1,strides=1, padding = 'same')(x)

	x = layers.UpSampling2D(size=(2, 2))(x)

	if model_args['n_out'] == 32:
		x = layers.UpSampling2D(size=(2, 2))(x)
	if model_args['n_out'] == 16:
		for _ in range(2):
			x = layers.UpSampling2D(size=(2, 2))(x)
	if model_args['n_out'] == 8:
		for _ in range(3):
			x = layers.UpSampling2D(size=(2, 2))(x)

	if model_args['n_out'] == 4:
		for _ in range(4):
			x = layers.UpSampling2D(size=(2, 2))(x)

	x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
	forward_mdl.trainable = False
	x = forward_mdl(x, training=False)

	return models.Model(inputs, x, name = "transformation_network")

def avg_downsample_network(forward_mdl, inputs,red_dim = 64):

	x = layers.AveragePooling2D(pool_size=(2, 2),strides=None, padding = 'valid')(inputs)

	if red_dim == 32:
		x = layers.AveragePooling2D(pool_size=(2, 2),strides=None, padding = 'valid')(x)
		x = layers.UpSampling2D(size=(2, 2))(x)
	if red_dim == 16:
		for _ in range(2):
			x = layers.AveragePooling2D(pool_size=(2, 2),strides=None, padding = 'valid')(x)
		for _ in range(2):
			x = layers.UpSampling2D(size=(2, 2))(x)
	if red_dim == 8:
		for _ in range(3):
			x = layers.AveragePooling2D(pool_size=(2, 2),strides=None, padding = 'valid')(x)
		for _ in range(3):
			x = layers.UpSampling2D(size=(2, 2))(x)
	if red_dim == 4:
		for _ in range(4):
			x = layers.AveragePooling2D(pool_size=(2, 2),strides=None, padding = 'valid')(x)

		for _ in range(4):
			x = layers.UpSampling2D(size=(2, 2))(x)

	x = layers.UpSampling2D(size=(2, 2))(x)

	forward_mdl.trainable = False
	x = forward_mdl(x, training=False)

	return models.Model(inputs, x, name = "transformation_network")

def general_downsample_network(forward_mdl, inputs,red_dim = 64,dim_method='bilinear'):

	x = tf.image.resize(images=inputs, size = [red_dim,red_dim],method=dim_method)
	x = layers.UpSampling2D(size=(2, 2))(x)
	if red_dim == 32:
		x = layers.UpSampling2D(size=(2, 2))(x)

	x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
	forward_mdl.trainable = False
	x = forward_mdl(x, training=False)

	return models.Model(inputs, x, name = "transformation_network")






def avg_downsample_network_with_contrast(forward_mdl, inputs,red_dim = 64,alpha = 1):

	x = inputs
	x = layers.AveragePooling2D(pool_size=(2, 2),strides=None, padding = 'valid')(x)

	if red_dim == 32:
		x = layers.AveragePooling2D(pool_size=(2, 2),strides=None, padding = 'valid')(x)

		if alpha is not None:		
			x = tf.image.adjust_contrast(x, alpha)
		

		
		x = layers.UpSampling2D(size=(2, 2))(x)
	x = layers.UpSampling2D(size=(2, 2))(x)

	x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1) # shape = [batch, height, width, channels]
		
	forward_mdl.trainable = False
	x = forward_mdl(x, training=False)

	return models.Model(inputs, x, name = "transformation_network")

