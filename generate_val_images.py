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
from PIL import Image
import utils,config
from tensorflow.keras import  layers,optimizers
import custom_model,custom_metrics


import tensorflow as tf

import utils
from cnn_models import create_ecker_cnn_model

def find_best_run(load_dir, n_out):
	folders = [ name for name in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, name)) ]
	last_run = folders[-1]
	last_run_dir = os.path.join(load_dir,last_run)
	tracking_df = pd.read_csv(os.path.join(last_run_dir,'track.csv'))


	best_run = tracking_df['test_neuronal_cc'].where((tracking_df['n_out'] == n_out) & (tracking_df['n_channel'] == 1000)).idxmax(0)
    


	return best_run



__basename__ = os.path.basename(__file__)
__name__, _ = os.path.splitext(__basename__)
__time__ = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# print(__basename__,__name__,__time__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # i.e. keep all message

experiment_directory = r'F:\Retina_project\Dataset_public\models\forward_model'
dataset_folder = r'F:\Retina_project\Dataset_public\models\forward_model'

gpus = tf.config.list_physical_devices(device_type='GPU')
if gpus:
	is_gpu_available = True
else:
	is_gpu_available = False

wn_files,files_highres ,sta_files,flash_file = config.get_file_numbers_actor_highres()
# stim_path, spikes_root, result_save_root, data_save_root = utils.get_file_path('alien','none','actor_high_resolution_400ms_stim')
stim_path = r'F:\Retina_project\Dataset_public\stimulus\high_res_stim'
spikes_root = r'F:\Retina_project\Dataset_public\spikes_data\high_res_spikes.npy'

train_x,val_x,test_x,train_y,val_y,test_y = utils.load_train_val_test(dataset_folder)
train_y = utils.merge_test_y(train_y)
val_y = utils.merge_test_y(val_y)



#Initializing and trained forward model
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

model = create_ecker_cnn_model(model_kwargs=model_kwargs, train_data=train_data, name="model")
model_handler = model.create_handler(
    directory=experiment_directory,
    train_data=(train_x, train_y),
    val_data=(val_x, val_y),
    test_data=(test_x, test_y),
)

run_name = "best_run"
model_handler.load(run_name=run_name)


model_args = {}
model_args['n_channel'] = 2
model_args['kernal_size']=2
model_args['l2_reg'] = 1e-3
model_args['n_out'] = 64

experiment_directory = r'F:\Retina_project\Dataset_public\test_pub\babak_actor_training_with_new_loss'


n_out=32
run_num = find_best_run(experiment_directory,n_out = n_out)
print('best run number', run_num)

run_name = "run_{:03d}".format(run_num)
load_dir =os.path.join(experiment_directory,run_name)
model_args = utils.load_params_actor(load_dir, model_args)

load_dir =os.path.join(experiment_directory,run_name)
model_args = utils.load_params_actor(load_dir, model_args)
actor_network = custom_model.babak_img_transformation_network(model,model_args,alpha=0.01,load_actor=True)
# actor_network.build(input_shape = (None,128,128,1))
# print(actor_network.summary())
# actor_network.compile(loss='poisson', optimizer=optimizers.Adam(0.002), metrics=[custom_metrics.cc_met,custom_metrics.rmse_met, custom_metrics.fev_met])

actor_network.load_weights(os.path.join(load_dir,'my_model_checkpoint')).expect_partial()
# actor_network.build(input_shape = (None,128,128,1))

dummy_input = tf.ones((1, 128,128,1))
actor_network(dummy_input)
print(actor_network(dummy_input))
print(actor_network.summary())





#

model_args['n_out'] = n_out
avg_network = custom_model.avg_downsample_network(model,forward_inp,red_dim = model_args['n_out'])
# avg_network = custom_model.ricker_downsample_with_contrast(model,forward_inp,red_dim = model_args['n_out'],alpha=1.,sigma=0.6,kappa = None,beta = 1)
avg_network.compile(loss='poisson', optimizer=optimizers.Adam(0.002), metrics=[custom_metrics.cc_met,custom_metrics.rmse_met, custom_metrics.fev_met])

# utils.plot_paired_test(fwd_model,avg_network,actor_network,test_x,test_y) 
layer_track = [ 'up','average','clip']

load_img = val_x #had to use 100 from val_x and 100 from test_x
n_image=100
actor_images,orig_images= utils.visualize_transformation_hardcode(actor_network,load_img,layer_track,image_save = load_dir,transform_type='actor',n_image=n_image)
avg_images,orig_images = utils.visualize_transformation(avg_network,load_img,layer_track,image_save = load_dir,transform_type='average',n_image=n_image)
# utils.plot_paired_test(fwd_model,avg_network,actor_network,val_x,val_y, save_dir = load_dir, nout=n_out) 

print('best run number', run_num) 
for i in range(len(actor_images)):
    orig_image = np.squeeze(np.array(orig_images[i]))
    assert np.max(orig_image) <= 1
    assert np.max(orig_image) >= 0

    act_image = np.squeeze(np.array(actor_images[i]))
    assert np.max(act_image) <= 1
    assert np.max(act_image) >= 0

    avg_image = np.squeeze(np.array(avg_images[i]))
    assert np.max(avg_image) <= 1
    assert np.max(avg_image) >= 0

    image_diff = np.abs(act_image - avg_image)
    image_dir = os.path.join(load_dir,'visualize_transformation_difference')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    save_dir = os.path.join(image_dir,'image_%d'%(i))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    orig_image = orig_image*255
    img = Image.fromarray(orig_image)
    img = img.convert("L")
    img.save(os.path.join(save_dir,  '{}.png'.format('orig_image')))
    act_image = act_image*255
    img = Image.fromarray(act_image)
    img = img.convert("L")
    img.save(os.path.join(save_dir,  '{}.png'.format('act_image')))
    avg_image = avg_image*255
    img = Image.fromarray(avg_image)
    img = img.convert("L")
    img.save(os.path.join(save_dir,  '{}.png'.format('avg_image')))


load_img = val_x #had to use 100 from val_x and 100 from test_x
n_image=100
actor_images,orig_images= utils.visualize_transformation_hardcode(actor_network,load_img,layer_track,image_save = load_dir,transform_type='actor',n_image=n_image)
avg_images,orig_images = utils.visualize_transformation(avg_network,load_img,layer_track,image_save = load_dir,transform_type='average',n_image=n_image)
# utils.plot_paired_test(fwd_model,avg_network,actor_network,val_x,val_y, save_dir = load_dir, nout=n_out) 

print('best run number', run_num) 
for i in range(len(actor_images)):
    orig_image = np.squeeze(np.array(orig_images[i]))
    assert np.max(orig_image) <= 1
    assert np.max(orig_image) >= 0

    act_image = np.squeeze(np.array(actor_images[i]))
    assert np.max(act_image) <= 1
    assert np.max(act_image) >= 0

    avg_image = np.squeeze(np.array(avg_images[i]))
    assert np.max(avg_image) <= 1
    assert np.max(avg_image) >= 0

    image_diff = np.abs(act_image - avg_image)
    image_dir = os.path.join(load_dir,'visualize_transformation_difference')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    save_dir = os.path.join(image_dir,'image_%d'%(i+100))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    orig_image = orig_image*255
    img = Image.fromarray(orig_image)
    img = img.convert("L")
    img.save(os.path.join(save_dir,  '{}.png'.format('orig_image')))
    act_image = act_image*255
    img = Image.fromarray(act_image)
    img = img.convert("L")
    img.save(os.path.join(save_dir,  '{}.png'.format('act_image')))
    avg_image = avg_image*255
    img = Image.fromarray(avg_image)
    img = img.convert("L")
    img.save(os.path.join(save_dir,  '{}.png'.format('avg_image')))
