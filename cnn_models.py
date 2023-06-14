#code adapted from Goldin et al. 2022. For original code, please refer to theirs.

import copy
import logging
import matplotlib.patches as pcs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf

from SpatialXFeatureJointL1Readout import SpatialXFeatureJointL1Readout
from StackedConv2DCore import StackedConv2DCore
from lstmconv2d import ConvLSTM2d

from learning_rate_on_plateau import CustomReduceLearningRateOnPlateauCallback
from utils import corrcoef,deepupdate
# from goldin_checkpoint import CustomModelCheckpointCallback

import custom_metrics
import utils

logger = logging.getLogger(__name__)

# Model.
class neuronal_CC_Callback(tf.keras.callbacks.Callback):
	def __init__(self,X_train,y_train,X_val,y_val,running_track=None):

		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.train_track = running_track['train_neu_cc']
		self.val_track = running_track['val_neu_cc']
		self.train_rmse_track = running_track['train_rmse']
		self.val_rmse_track = running_track['val_rmse']
	  # def on_train_begin(self, logs={}):

	  #only do it once instead of every epoch to save time. After finding best model can use on_epoch_end to get curve
	# def on_epoch_end(self, epoch, logs={}):
	#     train_cc,train_rmse = utils.plot_neuron_response(self.model,self.X_train,self.y_train)
	#     val_cc,val_rmse = utils.plot_neuron_response(self.model,self.X_val,self.y_val)
	#     plt.close('all')

	#     self.train_track.append(train_cc)
	#     self.val_track.append(val_cc)
	#     self.train_rmse_track.append(train_rmse)
	#     self.val_rmse_track.append(val_rmse)

	#     print('neuronal_CC: ',val_cc)
	#     print('val_rmse: ',val_rmse)
	def on_train_end(self, epoch, logs={}):
		train_cc,train_rmse = utils.plot_neuron_response(self.model,self.X_train,self.y_train)
		val_cc,val_rmse = utils.plot_neuron_response(self.model,self.X_val,self.y_val)
		plt.close('all')

		self.train_track.append(train_cc)
		self.val_track.append(val_cc)
		self.train_rmse_track.append(train_rmse)
		self.val_rmse_track.append(val_rmse)

		print('neuronal_CC: ',val_cc)
		print('val_rmse: ',val_rmse)

class RegularCNNModel(tf.keras.Model):
	"""Regular CNN model."""

	model_default_kwargs = {
		"core": {
			"nbs_kernels": (4,),
			"kernel_sizes": (21,),
			"strides": (1,),
			"paddings": ('valid',),
			"dilation_rates": (1,),
			"activations": ('relu',),
			"smooth_factors": (0.001,),  # (0.01,),
			"sparse_factors": (None,),
			"name": 'core',
		},
		"readout": {
			"nb_cells": 39,  # TODO correct!
			"x": None,
			"y": None,
			"spatial_masks_initializer": 'truncated normal',  # TODO correct!
			"feature_weights_initializer": 'truncated normal',  # TODO correct!
			"non_negative_feature_weights": True,  # TODO try `True` instead?
			"spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
			"feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
			"name": 'readout',
		}
	}

	def __init__(
			self, model_kwargs=None, learning_rate=0.002, train_data=None, name="model",running_track=None, **kwargs
	):
		"""Initialization of the model."""

		super().__init__(name=name, **kwargs)

		# Model keyword arguments.
		self.model_kwargs = copy.deepcopy(self.model_default_kwargs)
		if self.model_kwargs is not None:
			self.model_kwargs = deepupdate(self.model_kwargs, model_kwargs)
		if train_data is not None:
			train_x, train_y = train_data
			self.model_kwargs = deepupdate(
				self.model_kwargs,
				{
					"readout": {
						"x": train_x,
						"y": train_y,
					}
				}
			)
		# ...
		self.learning_rate = learning_rate
		# Initialize core.
		core_kwargs = self.model_kwargs["core"]
		self.core = StackedConv2DCore(**core_kwargs)
		# Initialize readout.
		readout_kwargs = self.model_kwargs["readout"]
		self.readout = SpatialXFeatureJointL1Readout(**readout_kwargs)
		self._running_track = running_track

	def compile(self, **kwargs):
		"""Configure the learning process of the model."""

		if self._is_compiled:
			logger.warning("Model has already been compiled.")
		else:
			optimizer = tf.keras.optimizers.Adam(
				learning_rate=self.learning_rate,
				# beta_1=0.9,
				# beta_2=0.999,
				# epsilon=1e-07,
				# amsgrad=False,
				name='Adam',
				# **kwargs,
			)
			loss = tf.keras.losses.Poisson(
				# reduction=losses_utils.ReductionV2.AUTO,
				name='poisson'
			)
			metrics = [
				tf.keras.metrics.Poisson(
					name='poisson'


					# dtype=None
				), custom_metrics.cc_met, 
					custom_metrics.rmse_met, 
					custom_metrics.fev_met, 
					custom_metrics.norm_rmse_met
			]
			super().compile(
				optimizer=optimizer,
				loss=loss,
				metrics=metrics,
				# loss_weights=None,
				# sample_weight_mode=None,
				# weighted_metrics=None,
				# target_tensors=None,
				# distribute=None,
				# **kwargs
			)

		return

	def call(self, inputs, training=False, **kwargs):
		"""Forward computation of the model."""

		internals = self.core(inputs, training=training)
		outputs = self.readout(internals)

		return outputs

	def create_handler(self, *args, **kwargs):

		return _RegularCNNModelHandler(self,*args, **kwargs)

create_regular_cnn_model = RegularCNNModel  # i.e. alias

class KlindtCNNModel(RegularCNNModel):

	model_default_kwargs = {
		"core": {
			"nbs_kernels": (4,),
			"kernel_sizes": (21,),
			"strides": (1,),
			"paddings": ('valid',),
			"dilation_rates": (1,),
			"activations": ('relu',),
			"smooth_factors": (0.001,),  # (0.01,),
			"sparse_factors": (None,),
			"name": 'core',
		},
		"readout": {
			"nb_cells": 39,  # TODO correct!
			"x": None,
			"y": None,
			"spatial_masks_initializer": '[Klindt et al., 2017]',
			"feature_weights_initializer": '[Klindt et al., 2017]',
			"non_negative_feature_weights": True,  # TODO try `True` instead?
			"spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
			"feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
			"name": 'readout',
		},
	}

	def __init__(self, *args, learning_rate=0.001, **kwargs):

		super().__init__(*args, learning_rate=learning_rate, **kwargs)


create_klindt_cnn_model = KlindtCNNModel  # i.e. alias

class EckerCNNModel(RegularCNNModel):

	model_default_kwargs = {
		"core": {
			"nbs_kernels": (4,),
			"kernel_sizes": (31,),
			"strides": (1,),
			"paddings": ('valid',),
			"dilation_rates": (1,),
			"activations": ('relu',),
			"smooth_factors": (0.001,),  # (0.01,),
			"sparse_factors": (None,),
			"name": 'core',
		},
		"readout": {
			"nb_cells": 39,  # TODO correct!
			"x": None,
			"y": None,
			"spatial_masks_initializer": '[Ecker et al., 2019]',
			"feature_weights_initializer": '[Ecker et al., 2019]',
			"non_negative_feature_weights": True,  # TODO try `True` instead?
			"spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
			"feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
			"name": 'readout',
		},
	}

	def __init__(self, *args, learning_rate=0.001, **kwargs):

		super().__init__(*args, learning_rate=learning_rate, **kwargs)


create_ecker_cnn_model = EckerCNNModel  # i.e. alias

class RegularRCNNModel(tf.keras.Model):
	"""Regular CNN model."""

	model_default_kwargs = {
		"core": {
			"nbs_kernels": (4,),
			"kernel_sizes": (21,),
			"strides": (1,),
			"paddings": ('valid',),
			"dilation_rates": (1,),
			"activations": ('relu',),
			"smooth_factors": (0.001,),  # (0.01,),
			"sparse_factors": (None,),
			"name": 'core',
		},
		"readout": {
			"nb_cells": 39,  # TODO correct!
			"x": None,
			"y": None,
			"spatial_masks_initializer": 'truncated normal',  # TODO correct!
			"feature_weights_initializer": 'truncated normal',  # TODO correct!
			"non_negative_feature_weights": True,  # TODO try `True` instead?
			"spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
			"feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
			"name": 'readout',
		}
	}

	def __init__(
			self, model_kwargs=None, learning_rate=0.002, train_data=None, name="model",running_track=None, **kwargs
	):
		"""Initialization of the model."""

		super().__init__(name=name, **kwargs)

		# Model keyword arguments.
		self.model_kwargs = copy.deepcopy(self.model_default_kwargs)
		if self.model_kwargs is not None:
			self.model_kwargs = deepupdate(self.model_kwargs, model_kwargs)
		if train_data is not None:
			train_x, train_y = train_data
			self.model_kwargs = deepupdate(
				self.model_kwargs,
				{
					"readout": {
						"x": train_x,
						"y": train_y,
					}
				}
			)
		# ...
		self.learning_rate = learning_rate
		# Initialize core.
		core_kwargs = self.model_kwargs["core"]
		self.core = ConvLSTM2d(**core_kwargs)
		# Initialize readout.
		readout_kwargs = self.model_kwargs["readout"]
		self.readout = SpatialXFeatureJointL1Readout(**readout_kwargs)
		self._running_track = running_track

	def compile(self, **kwargs):
		"""Configure the learning process of the model."""

		if self._is_compiled:
			logger.warning("Model has already been compiled.")
		else:
			optimizer = tf.keras.optimizers.Adam(
				learning_rate=self.learning_rate,
				# beta_1=0.9,
				# beta_2=0.999,
				# epsilon=1e-07,
				# amsgrad=False,
				name='Adam',
				# **kwargs,
			)
			loss = tf.keras.losses.Poisson(
				# reduction=losses_utils.ReductionV2.AUTO,
				name='poisson'
			)
			metrics = [
				tf.keras.metrics.Poisson(
					name='poisson'


					# dtype=None
				), custom_metrics.cc_met, 
					custom_metrics.rmse_met, 
					custom_metrics.fev_met, 
					custom_metrics.norm_rmse_met
			]
			super().compile(
				optimizer=optimizer,
				loss=loss,
				metrics=metrics,
				# loss_weights=None,
				# sample_weight_mode=None,
				# weighted_metrics=None,
				# target_tensors=None,
				# distribute=None,
				# **kwargs
			)

		return

	def call(self, inputs, training=False, **kwargs):
		"""Forward computation of the model."""

		internals = self.core(inputs, training=training)
		outputs = self.readout(internals)

		return outputs

	def create_handler(self, *args, **kwargs):

		return _RegularCNNModelHandler(self,*args, **kwargs)

create_regular_rcnn_model = RegularRCNNModel  # i.e. alias

class EckerRCNNModel(RegularRCNNModel):

	model_default_kwargs = {
		"core": {
			"nbs_kernels": (4,),
			"kernel_sizes": (31,),
			"strides": (1,),
			"paddings": ('valid',),
			"dilation_rates": (1,),
			"activations": ('relu',),
			"smooth_factors": (0.001,),  # (0.01,),
			"sparse_factors": (None,),
			"name": 'core',
		},
		"readout": {
			"nb_cells": 39,  # TODO correct!
			"x": None,
			"y": None,
			"spatial_masks_initializer": '[Ecker et al., 2019]',
			"feature_weights_initializer": '[Ecker et al., 2019]',
			"non_negative_feature_weights": True,  # TODO try `True` instead?
			"spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
			"feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
			"name": 'readout',
		},
	}

	def __init__(self, *args, learning_rate=0.001, **kwargs):

		super().__init__(*args, learning_rate=learning_rate, **kwargs)

create_ecker_rcnn_model = EckerRCNNModel  # i.e. alias


class RegulardeformCNNModel(tf.keras.Model):
	"""Regular CNN model."""

	model_default_kwargs = {
		"core": {
			"nbs_kernels": (4,),
			"kernel_sizes": (21,),
			"strides": (1,),
			"paddings": ('valid',),
			"dilation_rates": (1,),
			"activations": ('relu',),
			"smooth_factors": (0.001,),  # (0.01,),
			"sparse_factors": (None,),
			"name": 'core',
		},
		"readout": {
			"nb_cells": 39,  # TODO correct!
			"x": None,
			"y": None,
			"spatial_masks_initializer": 'truncated normal',  # TODO correct!
			"feature_weights_initializer": 'truncated normal',  # TODO correct!
			"non_negative_feature_weights": True,  # TODO try `True` instead?
			"spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
			"feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
			"name": 'readout',
		}
	}

	def __init__(
			self, model_kwargs=None, learning_rate=0.002, train_data=None, name="model",running_track=None, **kwargs
	):
		"""Initialization of the model."""

		super().__init__(name=name, **kwargs)

		# Model keyword arguments.
		self.model_kwargs = copy.deepcopy(self.model_default_kwargs)
		if self.model_kwargs is not None:
			self.model_kwargs = deepupdate(self.model_kwargs, model_kwargs)
		if train_data is not None:
			train_x, train_y = train_data
			self.model_kwargs = deepupdate(
				self.model_kwargs,
				{
					"readout": {
						"x": train_x,
						"y": train_y,
					}
				}
			)
		# ...
		self.learning_rate = learning_rate
		# Initialize core.
		core_kwargs = self.model_kwargs["core"]
		self.core = DeformConv2DCore(**core_kwargs)
		# Initialize readout.
		readout_kwargs = self.model_kwargs["readout"]
		self.readout = SpatialXFeatureJointL1Readout(**readout_kwargs)
		self._running_track = running_track

	def compile(self, **kwargs):
		"""Configure the learning process of the model."""

		if self._is_compiled:
			logger.warning("Model has already been compiled.")
		else:
			optimizer = tf.keras.optimizers.Adam(
				learning_rate=self.learning_rate,
				# beta_1=0.9,
				# beta_2=0.999,
				# epsilon=1e-07,
				# amsgrad=False,
				name='Adam',
				# **kwargs,
			)
			loss = tf.keras.losses.Poisson(
				# reduction=losses_utils.ReductionV2.AUTO,
				name='poisson'
			)
			metrics = [
				tf.keras.metrics.Poisson(
					name='poisson'


					# dtype=None
				), custom_metrics.cc_met, 
					custom_metrics.rmse_met, 
					custom_metrics.fev_met, 
					custom_metrics.norm_rmse_met
			]
			super().compile(
				optimizer=optimizer,
				loss=loss,
				metrics=metrics,
				# loss_weights=None,
				# sample_weight_mode=None,
				# weighted_metrics=None,
				# target_tensors=None,
				# distribute=None,
				# **kwargs
			)

		return

	def call(self, inputs, training=False, **kwargs):
		"""Forward computation of the model."""

		internals = self.core(inputs, training=training)
		outputs = self.readout(internals)

		return outputs

	def create_handler(self, *args, **kwargs):

		return _RegularCNNModelHandler(self,*args, **kwargs)

create_regular_rcnn_model = RegularRCNNModel  # i.e. alias

class EckerDeformCNNModel(RegulardeformCNNModel):

	model_default_kwargs = {
		"core": {
			"nbs_kernels": (4,),
			"kernel_sizes": (31,),
			"strides": (1,),
			"paddings": ('valid',),
			"dilation_rates": (1,),
			"activations": ('relu',),
			"smooth_factors": (0.001,),  # (0.01,),
			"sparse_factors": (None,),
			"name": 'core',
		},
		"readout": {
			"nb_cells": 39,  # TODO correct!
			"x": None,
			"y": None,
			"spatial_masks_initializer": '[Ecker et al., 2019]',
			"feature_weights_initializer": '[Ecker et al., 2019]',
			"non_negative_feature_weights": True,  # TODO try `True` instead?
			"spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
			"feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
			"name": 'readout',
		},
	}

	def __init__(self, *args, learning_rate=0.001, **kwargs):

		super().__init__(*args, learning_rate=learning_rate, **kwargs)

create_ecker_deformcnn_model = EckerDeformCNNModel  # i.e. alias


class _RegularCNNModelHandler:
	def __init__(self, model, directory=None, train_data=None, val_data=None, test_data=None,running_track=None):

		model.compile()

		self._model = model
		self._directory = directory
		self._train_data = train_data
		self._val_data = val_data
		self._test_data = test_data
		# self._running_track = running_track

		self._checkpoints = dict()
		self._tag = None
		self._run_name = None
	def _get_checkpoint_weights_path(self, tag='final', run_name=None):
		if self._directory is not None:
			# ...
			if run_name is not None:
				directory = os.path.join(self._directory, run_name)
			else:
				directory = self._directory
			# ...
			if tag is None:
				path = os.path.join(directory, "checkpoint_final")
			elif isinstance(tag, str):
				path = os.path.join(directory, "checkpoint_{}".format(tag))
			elif isinstance(tag, int):
				path = os.path.join(directory, "checkpoint_{:05d}".format(tag))
			else:
				raise TypeError("unexpected tag type: {}".format(type(tag)))
		else:
			path = None

		return path
	def train(self, train_data, val_data,epochs=None, is_gpu_available=False, run_name=None):
		train_x, train_y = train_data
		val_x, val_y = val_data

		# Evaluate model (to make the loading effective, to be able to save the initial weights).
		_ = self._model.evaluate(val_x, val_y, batch_size=32, verbose=0)
		monitor = 'val_poisson'  # i.e. not 'val_loss'

		callbacks = []

		# # # Enable learning rate decays.
		learning_rate_decay_factor = 0.5  # TODO use `0.1` instead (default)?
		# TODO set a `minimum_learning_rate` to limit the number of learning rate decays?
		callback = CustomReduceLearningRateOnPlateauCallback(
			monitor=monitor,
			factor=learning_rate_decay_factor,
			patience=10,
			verbose=1,  # {0 (quiet, default), 1 (update messages)}
			# mode='auto',  # {auto (default), min, max}
			# min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes
			# cooldown=0,  # number of epochs to wait before resuming normal operation after the learning rate has been reduced  # noqa
			# min_lr=0,  # lower bound on the learning rate (default: 0)
			restore_best_weights=True,  # TODO understand why TensorFlow does not implement this by default???
			# **kwargs,
		)
		callbacks.append(callback)
		callback = tf.keras.callbacks.EarlyStopping(
			monitor=monitor,
			# min_delta=0,
			patience=20,  # use `0` instead (default)?
			verbose=2,  # {0 (quiet?, default), 1 (update messages?)}
			# mode='auto',
			# baseline=None,
			restore_best_weights=True,
		)
		callbacks.append(callback)




		neuronal_cc_callback = neuronal_CC_Callback(train_data[0],train_data[1],val_data[0],val_data[1],self._model._running_track)
		callbacks.append(neuronal_cc_callback)



		batch_size = 32  # 128  # 256  # 32
		if epochs is None:
			epochs = 1000 if is_gpu_available else 10

		verbose = 2 if is_gpu_available else 2 # verbosity mode (0 = silent, 1 = progress bar (interactive environment), 2 = one line per epoch (production environment))  # noqa
		history = self._model.fit(
			train_x,
			train_y,
			batch_size=batch_size,
			epochs=epochs,
			verbose=verbose,
			callbacks=callbacks,
			# validation_split=0.0,
			validation_data=(val_x, val_y))

		# # Save final weights (if necessary).
		path = self._get_checkpoint_weights_path(tag='final', run_name=run_name)
		if path is not None:
			self._model.save_weights(
				path,
				# overwrite=True,
				save_format='tf',  # or 'h5'?
			)
			# self._model.save(
			#     path,
			#     # overwrite=True,
			#     save_format='tf',  # or 'h5'?
			# )

		# Update attributes.
		self._tag = 'final'
		self._run_name = run_name

		return history

	def _convert_domain(self, domain):

		from tensorboard.plugins.hparams import api as hp  # TODO move to top of file?

		if domain is None:
			values = ['None']
			dtype = None  # TODO correct or remove!
			converted_domain = hp.Discrete(values, dtype=dtype)
		elif isinstance(domain, (int, float, str)):
			values = [domain]
			dtype = None  # TODO correct of remove!
			converted_domain = hp.Discrete(values, dtype=dtype)
		elif isinstance(domain, tuple):
			converted_domain = tuple([
				self._convert_domain(sub_domain)
				for sub_domain in domain
			])  # TODO avoid `ValueError: not a domain: (RealInterval(0.002, 0.04), RealInterval(0.002, 0.04))`!
		elif isinstance(domain, set):
			values = list(domain)
			dtype = None  # TODO correct or remove!
			converted_domain = hp.Discrete(values, dtype=dtype)
		elif isinstance(domain, list):
			assert len(domain) == 2, domain
			min_value = domain[0]
			max_value = domain[1]
			if isinstance(min_value, float) and isinstance(max_value, float):
				assert min_value <= max_value, domain
				converted_domain = hp.RealInterval(min_value=min_value, max_value=max_value)
			elif isinstance(min_value, int) and isinstance(max_value, int):
				assert min_value <= max_value, domain
				converted_domain = hp.IntInterval(min_value=min_value, max_value=max_value)
			else:
				raise TypeError(
					"unexpected min_value ({}) and max_value types({})".format(type(min_value), type(max_value))
				)
		else:
			# TODO correct!
			raise TypeError("unexpected domain type ({})".format(type(domain)))

		return converted_domain
	def _convert_hyperparameters(self, hyperparameters):
		"""Hyperparameters conversion (from dict to TensorBoard API)"""

		from tensorboard.plugins.hparams import api as hp  # TODO move to top of file?

		assert isinstance(hyperparameters, dict), hyperparameters

		converted_hyperparameters = dict()
		for name, domain in hyperparameters.items():
			converted_domain = self._convert_domain(domain)
			if isinstance(converted_domain, tuple):
				for k, converted_sub_domain in enumerate(converted_domain):
					assert not isinstance(converted_sub_domain, tuple)  # TODO implement?
					sub_name = name + "_{}".format(k)
					converted_hyperparameters[sub_name] = hp.HParam(sub_name, domain=converted_sub_domain)
			else:
				converted_hyperparameters[name] = hp.HParam(name, domain=converted_domain)

		return converted_hyperparameters
	def _sample_domain(self, domain):

		if domain is None:
			sampled_value = domain
		elif isinstance(domain, (int, float, str)):
			sampled_value = domain
		elif isinstance(domain, tuple):
			sampled_value = tuple([
				self._sample_domain(sub_domain)
				for sub_domain in domain
			])
		elif isinstance(domain, set):
			values = list(domain)
			sampled_value = random.choice(values)
		elif isinstance(domain, list):
			assert len(domain) == 2, domain
			min_value = domain[0]
			max_value = domain[1]
			if isinstance(min_value, float) and isinstance(max_value, float):
				assert min_value <= max_value, domain
				# sampled_value = random.uniform(min_value, max_value)  # i.e. uniform
				sampled_value = np.exp(random.uniform(np.log(min_value), np.log(max_value)))  # i.e. log-uniform
			elif isinstance(min_value, int) and isinstance(max_value, int):
				assert min_value <= max_value, domain
				sampled_value = random.randint(min_value, max_value)
			else:
				raise TypeError(
					"unexpected min_value ({}) and max_value types({})".format(type(min_value), type(max_value))
				)
		else:
			# TODO correct!
			raise TypeError("unexpected domain type ({})".format(type(domain)))

		return sampled_value
	def _sample_hyperparameters(self, hyperparameters, seed=None):  # TODO move outside class?

		assert isinstance(hyperparameters, dict), hyperparameters

		if seed is None:
			# random.seed(a=None)  # i.e. use current system time to initialize the random number generator.
			pass  # TODO correct?
		else:
			random.seed(a=seed)

		sampled_hyperparameters = dict()
		for name, domain in hyperparameters.items():
			sampled_value = self._sample_domain(domain)
			sampled_hyperparameters[name] = sampled_value

		return sampled_hyperparameters
	def randomized_search(self, hyperparameters, train_data, val_data, test_data,is_gpu_available=False, nb_runs=2,tracking_dict = None):
		"""Hyperparameter optimization/tuning."""

		from tensorboard.plugins.hparams import api as hp  # TODO move to top of file?

		if self._directory is not None:
			tensorboard_path = os.path.join(self._directory, "logs")  # TODO move to class.
			hparams_summary_path = tensorboard_path  # TODO rename?
		else:
			hparams_summary_path = None

		if not os.path.isdir(hparams_summary_path):  # i.e. search has not already been ran

			if hparams_summary_path is not None:
				# Log the experiment configuration to TensorBoard.
				converted_hyperparameters = self._convert_hyperparameters(hyperparameters)
				with tf.summary.create_file_writer(hparams_summary_path).as_default():
					hp.hparams_config(
						hparams=list(converted_hyperparameters.values()),
						metrics=[
							hp.Metric('val_loss', display_name='val_loss'),
							hp.Metric('val_poisson', display_name='val_poisson'),
						],
						# time_created_secs=None,  # i.e. current time (default)
					)

			for run_nb in range(0, nb_runs):

				run_name = "run_{:03d}".format(run_nb)
				running_track = {'train_neu_cc':[],
								 'val_neu_cc':[],
								 'train_rmse':[],
								 'val_rmse':[]}


				# Sample a random combination of hyperparameters.
				sampled_hyperparameters = self._sample_hyperparameters(hyperparameters, seed=run_nb)
				# Sanity prints.
				print("Run {:03d}/{:03d}:".format(run_nb, nb_runs))
				for name, value in sampled_hyperparameters.items():
					print("    {}: {}".format(name, value))

				# Create model.
				model_kwargs = copy.deepcopy(self._model.model_kwargs)
				# # Update hyperparameters involved in this random search.
				for name, value in sampled_hyperparameters.items():
					keys = name.split('/')
					kwargs = model_kwargs
					for key in keys[:-1]:
						kwargs = kwargs[key]
					kwargs[keys[-1]] = value
				# # Clear TF graph (i.e. use same namespace).
				tf.keras.backend.clear_session()
				# # Instantiate & compile model.
				model = self._model.__class__(
					model_kwargs=model_kwargs,
					train_data=self._train_data,
					name="model",
					running_track = running_track
				)
				model.compile()
				self._model = model

				# Train model.
				history = self.train(train_data, val_data, is_gpu_available=is_gpu_available, run_name=run_name)

				if hparams_summary_path is not None:
					# run_summary_path = os.path.join(hparams_summary_path, run_name)
					run_summary_path = os.path.join(hparams_summary_path, run_name, "train")
					# Log the hyperparameters and metrics to TensorBoard.
					with tf.summary.create_file_writer(run_summary_path).as_default():
						# Log hyperparameter values for the current run/trial.
						formatted_hyperparameters = dict()
						for name, value in sampled_hyperparameters.items():
							if value is None:
								formatted_value = 'None'
								formatted_hyperparameters[name] = formatted_value
							elif isinstance(value, tuple):
								for k, sub_value in enumerate(value):
									sub_name = name + "_{}".format(k)
									if sub_value is None:
										formatted_value = 'None'
										formatted_hyperparameters[sub_name] = formatted_value
									else:
										formatted_hyperparameters[sub_name] = sub_value
							else:
								formatted_hyperparameters[name] = value
						_ = hp.hparams(
							formatted_hyperparameters,
							trial_id=run_name,
							# start_time_secs=None,  # i.e. current time
						)
						# Log hyperparameters for programmatic use.
						for name, value in formatted_hyperparameters.items():
							name = "hyperparameters/{}".format(name)
							value = value.item()  # i.e. numpy.float to float
							tf.summary.scalar(name, value, step=0, description=None)
						# Log metrics.
						for step, val_loss in enumerate(history.history['val_loss']):
							tf.summary.scalar('val_loss', val_loss, step=step)
						for step, val_poisson in enumerate(history.history['val_poisson']):
							tf.summary.scalar('val_poisson', val_poisson, step=step)
						# TODO use callbacks instead of writing these directly?

				
				results = self._model.evaluate(x=utils.merge_test_y(test_data[0]), y= utils.merge_test_y(test_data[1]), batch_size=32, return_dict = True,verbose = 2)
				
				if tracking_dict is not None:
					for metric,value in results.items():
						tracking_dict[metric].append(value)
				# running_track = (tracking_dict['train_neu_cc'],tracking_dict['val_neu_cc'],tracking_dict['train_nccc'],tracking_dict['val_nccc'],tracking_dict['train_rmse'],tracking_dict['test_rmse'])
				running_track_lst = [running_track['train_neu_cc'],running_track['val_neu_cc'],None,None,running_track['train_rmse'],running_track['val_rmse']]
				data = [train_data[0],train_data[1], val_data[0], val_data[1],test_data[0],test_data[1]]


				tracking_save_dir = os.path.join(hparams_summary_path, run_name)

				utils.plot_metrics(history,results,save_dir = tracking_save_dir)
				print(tracking_save_dir)
				train_avg_cc,val_avg_cc,test_avg_cc,train_avg_rmse,val_avg_rmse,test_avg_rmse,test_nc_cc = utils.plot_CCs(model, data,running_track_lst,tracking_save_dir,None)
				
				tracking_dict['train_neuronal_rmse'].append(train_avg_rmse)
				tracking_dict['val_neuronal_rmse'].append(val_avg_rmse)
				tracking_dict['test_neuronal_rmse'].append(test_avg_rmse)



				tracking_dict['smooth_factors'].append(model_kwargs['core']['smooth_factors'])
				tracking_dict['spatial_sparsity_factor'].append(model_kwargs['readout']['spatial_sparsity_factor'])
				tracking_dict['feature_sparsity_factor'].append(model_kwargs['readout']['feature_sparsity_factor'])


				tracking_dict['train_neuronal_cc'].append(train_avg_cc)
				tracking_dict['val_neuronal_cc'].append(val_avg_cc)
				tracking_dict['test_neuronal_cc'].append(test_avg_cc)
				tracking_dict['test_nc_cc'].append(test_nc_cc)


				tracking_df = pd.DataFrame(tracking_dict)
				tracking_df.to_csv(os.path.join(tracking_save_dir,"track.csv"))

				if tracking_dict['test_neuronal_cc'][-1] == max(tracking_dict['test_neuronal_cc']):
					self.best_model = self._model
					self.best_run = run_nb
		else:

			logger.debug("randomized search has already been ran")


		return self.best_model, tracking_dict


	def load(self, tag=None, run_name=None):
		"""Load model."""

		# Load trained weights (if possible).
		try:
			self._model.load_weights(
				self._get_checkpoint_weights_path(tag=tag, run_name=run_name),
				# by_name=False,
				# skip_mismatch=False,
			).expect_partial()
			# c.f. https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-objec.  # noqa
			self._tag = tag
			self._run_name = run_name
		except (tf.errors.NotFoundError, ValueError):
			raise FileNotFoundError

		# Test modtl (if possible, to make the loading effective).
		if self._test_data is not None:
			test_x, test_y = self._test_data
			_ = self._model.evaluate(test_x, test_y, batch_size=32, verbose=0)
		else:
			raise NotImplementedError("TODO")

		return