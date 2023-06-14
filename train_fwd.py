import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import gc
import itertools
from collections import defaultdict
import math
import datetime
# import logging
from collections import defaultdict

import tensorflow as tf
import utils
import config
from cnn_models import create_ecker_cnn_model

__basename__ = os.path.basename(__file__)
__name__, _ = os.path.splitext(__basename__)
__time__ = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # i.e. keep all message
gpus = tf.config.list_physical_devices(device_type='GPU')
if gpus:
    is_gpu_available = True
else:
    is_gpu_available = False

#directory to save model
experiment_directory = r'F:\Dataset_public\test_pub\forward_training'

wn_files,files_highres ,sta_files,flash_file = config.get_file_numbers_actor_highres()
stim_path = r'F:\Dataset_public\stimulus\high_res_stim'
spikes_root = r'F:\Dataset_public\spikes_data\high_res_spikes.npy'


spikes_data = np.load(spikes_root) #nfiles, nNeurons, bins
print(spikes_data.shape)
wn_resp, nat_resp = utils.sum_response(spikes_data, 40, (wn_files,files_highres),num_unique_image=200)
wn_stim,nat_stim = utils.get_stim(stim_path,(wn_files,files_highres),num_unique_image=200,num_images=24000)
neuron_pass = utils.stability_check(wn_resp,wn_files,stability_thresh=0.3,num_unique_image=200)

# nat_resp = utils.avg_response_repeats(nat_resp,files_highres,num_unique_image=200)
nat_stim = utils.avg_nat_stim(nat_stim,files_highres,num_unique_image=200)

#split odd and even to facilitate noise corrected CC later on
nat_resp_odd,nat_resp_even = utils.avg_response_odd_even(nat_resp,files_highres,num_unique_image=200)
stim_train,stim_val,stim_test,nat_train_odd,nat_val_odd,nat_test_odd,nat_train_even,nat_val_even,nat_test_even = utils.train_val_test_split_odd_even(nat_stim,nat_resp_odd,nat_resp_even,0.1,0.1,neuron_pass=neuron_pass)


#massaging data format
train_x = np.expand_dims(stim_train, 3)
val_x = np.expand_dims(stim_val, 3)
test_x = np.expand_dims(np.concatenate([stim_test,stim_test]), 3)

train_y = np.concatenate([nat_train_odd,nat_train_even])
val_y = np.concatenate([nat_val_odd,nat_val_even])
test_y = np.concatenate([nat_test_odd,nat_test_even])

#save train,val,test data to load when training actor model to be consistent
utils.save_train_val_test(experiment_directory, train_x,val_x,test_x,train_y,val_y,test_y)

train_y = utils.merge_test_y(train_y)
val_y = utils.merge_test_y(val_y)

train_data = (train_x, train_y)
val_data = (val_x, val_y)
test_data = (test_x,test_y)

# Create model.
model_kwargs = {
    "core": {
        "nbs_kernels": (4,),
        "kernel_sizes": (7,), #can only be odd number, can try 21 as well
        "strides": (1,),
        "paddings": ('valid',), 
        "dilation_rates": (1,),
        "activations": ('softplus',), 
        "smooth_factors": (0.0033049721348414845,), 
        "sparse_factors": (None,),
        "name": 'core',
    },
    "readout": {
        'nb_cells':train_y.shape[1],
        "spatial_sparsity_factor": 0.00278407095321502,
        "feature_sparsity_factor": 1.33654461623723E-06,
        "name": 'readout',
    },
}


model = create_ecker_cnn_model(model_kwargs=model_kwargs, train_data=train_data, name="model")

# Get model handler.
model_handler = model.create_handler(
    directory=experiment_directory,
    train_data=(train_x, train_y),
    val_data=(val_x, val_y),
    test_data=(test_x, test_y),
)

hyperparameters = {
    'core/smooth_factors': ([1.0e-7, 1.0e+1],),
    # 'core/sparse_factors': ([1.0e-6, 1.0e+1],),
    'readout/spatial_sparsity_factor': [1.0e-7, 1.0e-1],
    'readout/feature_sparsity_factor': [1.0e-6, 1.0e0],
}  # TODO correct?

tracking_dict = defaultdict(list)
best_model, tracking_dict = model_handler.randomized_search(
    hyperparameters,
    train_data,
    val_data,
    test_data,
    is_gpu_available=is_gpu_available,
    nb_runs=1000,
    tracking_dict = tracking_dict
)