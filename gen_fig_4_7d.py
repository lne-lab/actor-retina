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
from pathlib import Path
from PIL import Image
import seaborn as sns
import scipy
from scipy.stats import ttest_rel,wilcoxon, shapiro, mannwhitneyu



__basename__ = os.path.basename(__file__)
__name__, _ = os.path.splitext(__basename__)
__time__ = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # i.e. keep all message

experiment_directory = r'F:\Retina_project\Dataset_public\models\forward_model'
dataset_folder = r'F:\Retina_project\Dataset_public\models\forward_model'

gpus = tf.config.list_physical_devices(device_type='GPU')
if gpus:
	is_gpu_available = True
else:
	is_gpu_available = False


train_x,val_x,test_x,train_y,val_y,test_y = utils.load_train_val_test(dataset_folder)
forward_inp = layers.Input(shape = train_x.shape[1:])
train_data = (train_x, train_y)
val_data = (val_x, val_y)
test_data = (test_x,test_y)
val_test_x = test_x
val_test_y = test_y



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
model_args['n_out'] = 32

experiment_directory = r'F:\Retina_project\Dataset_public\models\actor_model'
final_fig_root = Path(r'F:\Retina_project\Dataset_public\figures\figure_4_7d')


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


layer_track = ['conv', 'up','average','clip']
downsampling_methods_original =['average','bilinear','nearest','lanczos3','lanczos5','bicubic','gaussian','area','mitchellcubic']
downsampling_methods =['actor','average','average_with_contrast','bilinear','nearest','lanczos3','lanczos5','bicubic','gaussian','area','mitchellcubic']

transformed_img = {}



def calc_local_contrast(img,window):
	mean_op = np.ones((window,window))/(window*window)
	mean_of_sq = scipy.signal.convolve2d(img**2,mean_op,mode='same',boundary='symm')
	sq_of_mean = scipy.signal.convolve2d(img,mean_op,mode='same',boundary='symm')**2
	win_var = mean_of_sq - sq_of_mean
	return np.mean(win_var)

def calc_rmse(orig,downsampled):
	return np.sqrt(np.mean((orig-downsampled)**2))

def get_contrast_list(model,n_image,tf_type,load_img,layer_track):
	downsampled_images, orig_image = utils.visualize_transformation(model,load_img,layer_track,image_save = load_dir,transform_type=tf_type,n_image=n_image)

	contrast_list = []

	for i in range(n_image):
		contrast_list.append(calc_local_contrast(downsampled_images[i],7))
	return contrast_list



n_image_viz=val_test_x.shape[0]
# n_image_viz=3 #for testing

actor_contrast_lst = get_contrast_list(actor_network,n_image_viz,'actor',val_test_x,layer_track)


all_contrast_diff_to_act = {}
all_diff_result = {}

all_p_val_constrast_to_act = {}
all_p_val_NR_to_act = {}


for dim_method in downsampling_methods:
	
	final_fig_dir = final_fig_root/'methods_transformation'/dim_method
	os.makedirs(final_fig_dir,exist_ok=True)
	if dim_method == 'actor':
		avg_network = actor_network
	elif dim_method == 'average':
		avg_network = custom_model.avg_downsample_network(fwd_model,forward_inp,red_dim = model_args['n_out'])
	elif dim_method == 'average_with_contrast':
		avg_network = custom_model.avg_downsample_network_with_contrast(fwd_model,forward_inp,red_dim = model_args['n_out'],alpha=1.5)
	else:
		avg_network = custom_model.general_downsample_network(fwd_model,forward_inp,red_dim = model_args['n_out'],dim_method=dim_method)

	avg_network.compile(loss='poisson', optimizer=optimizers.Adam(0.002), metrics=[custom_metrics.cc_met,custom_metrics.rmse_met, custom_metrics.fev_met])

	downsampled_contrast_lst = get_contrast_list(avg_network,n_image_viz,dim_method,val_test_x,layer_track)

	if dim_method != 'actor':
		print(dim_method)
		perc_increase,diff_results = utils.plot_paired_test(fwd_model,avg_network,actor_network,val_test_x,val_test_y, save_dir = final_fig_dir,nout=n_out) 
		all_diff_result[dim_method] = diff_results['act - avg']

		all_contrast_diff_to_act[f'{dim_method}'] = np.array(actor_contrast_lst) - np.array(downsampled_contrast_lst)

		_,all_p_val_NR_to_act[f'{dim_method}'] = scipy.stats.wilcoxon(diff_results['fwd - act'],diff_results['fwd - avg'])  
		_,all_p_val_constrast_to_act[f'{dim_method}'] = scipy.stats.wilcoxon(actor_contrast_lst,downsampled_contrast_lst)        
	


all_diff_result_df = pd.DataFrame(all_diff_result)
WIDTH_SIZE=5
HEIGHT_SIZE=9


fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
sns.boxplot(all_diff_result_df[['average','average_with_contrast']])
plt.ylabel('Difference in reliability between average and enhanced average to actor downsampled images')
# plt.ylabel('Difference in neuronal reliability')
plt.tight_layout()
plt.savefig(final_fig_root/'methods_transformation'/f'diff_reliability_avg_to_act.svg')

fig = plt.figure(figsize=(HEIGHT_SIZE,WIDTH_SIZE))
sns.boxplot(all_diff_result_df[downsampling_methods_original])
plt.ylabel('Difference in reliability to actor downsampled images')
# plt.ylabel('Difference in neuronal reliability')
plt.tight_layout()
plt.savefig(final_fig_root/'methods_transformation'/f'diff_reliability_all_to_act.svg')



source_data = pd.DataFrame({'methods': all_p_val_NR_to_act.keys(), 'NR to actor': all_p_val_NR_to_act.values(),'contrast to actor': all_p_val_constrast_to_act.values()})
source_data.to_csv(final_fig_root/'methods_transformation'/f'p_val_NR_Cont_all_to_act.csv')





