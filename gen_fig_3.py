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
import logging
from collections import defaultdict

import tensorflow as tf
import utils,config

import custom_metrics
import custom_model
from tensorflow.keras import  layers,optimizers
from cnn_models import create_ecker_cnn_model

__basename__ = os.path.basename(__file__)
__name__, _ = os.path.splitext(__basename__)
__time__ = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # i.e. keep all message

experiment_directory = r'F:\Dataset_public\models\forward_model'
dataset_folder = r'F:\Dataset_public\models\forward_model'

gpus = tf.config.list_physical_devices(device_type='GPU')
if gpus:
    is_gpu_available = True
else:
    is_gpu_available = False


train_x,val_x,test_x,train_y,val_y,test_y = utils.load_train_val_test(dataset_folder)
forward_inp = layers.Input(shape = train_x.shape[1:])

model_kwargs = {
    "core": {
        "nbs_kernels": (4,),
        "kernel_sizes": (7,),
        "strides": (1,),
        "paddings": ('valid',),
        "dilation_rates": (1,),
        # "activations": ('relu',), 
        "activations": ('softplus',), 
        "smooth_factors": (0.001,),  
        "sparse_factors": (None,),
        "name": 'core',
    },
    "readout": {
        'nb_cells':train_y.shape[1],
        "spatial_sparsity_factor": 0.0001, 
        "feature_sparsity_factor": 0.1, 
        "name": 'readout',
    },
}

train_data = (train_x, train_y)
val_data = (val_x, val_y)
test_data = (test_x,test_y)

fwd_model = create_ecker_cnn_model(model_kwargs=model_kwargs, train_data=train_data, name="model")
model_handler = fwd_model.create_handler(
    directory=experiment_directory,
    train_data=(train_x, train_y),
    val_data=(val_x, val_y),
    test_data=(test_x, test_y),
)

run_name = "best_run"
model_handler.load(run_name=run_name)


#randomly initialize hyperparams and change later
model_args = {}
model_args['n_channel'] = 2
model_args['kernal_size']=2
model_args['l2_reg'] = 1e-3
model_args['n_out'] = 64

experiment_directory = r'F:\Dataset_public\models\actor_model'
final_fig_dir = r'F:\Dataset_public\figures\figure_3'
os.makedirs(final_fig_dir,exist_ok=True)

n_out=32
perc_increase_lst=[]
diff_df = []
n_out_order = []
act_avg_order = []
perc_increase_df = pd.DataFrame()

run_name = "best_run"
load_dir =os.path.join(experiment_directory,run_name)
model_args = utils.load_params_actor(load_dir, model_args)
actor_network = custom_model.img_transformation_network(fwd_model,forward_inp,train_y.shape[1],model_args)
actor_network.load_weights(os.path.join(load_dir,'my_model_checkpoint')).expect_partial()
avg_network = custom_model.avg_downsample_network(fwd_model,forward_inp,red_dim = model_args['n_out'])
avg_network.compile(loss='poisson', optimizer=optimizers.Adam(0.002), metrics=[custom_metrics.cc_met,custom_metrics.rmse_met, custom_metrics.fev_met])


perc_increase,diff_results = utils.plot_paired_test(fwd_model,avg_network,actor_network,test_x,test_y, save_dir = final_fig_dir,nout=n_out) 








