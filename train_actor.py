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
from learning_rate_on_plateau import CustomReduceLearningRateOnPlateauCallback


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



#initialize actor model hyperparameters
n_channel_lst = [1,2,4,6,8,16]
kernal_size_lst = [2,4,8,16,21,31]
l1_reg_lst = [0]
l2_reg_lst = [1e-1,1e-2,1e-3,1e-4]
n_out_lst = [4,8,16,32,64]

combination = list(itertools.product(n_channel_lst,
                                    kernal_size_lst,
                                    l1_reg_lst,
                                    l2_reg_lst,
                                    n_out_lst))

params_key = ['n_channel',
            'kernal_size',
            'l1_reg',
            'l2_reg',
            'n_out']

run_num = 0 #have to be 0
#directory for actor training
experiment_directory = r'F:\Dataset_public\test_pub\actor_training'
os.makedirs(experiment_directory,exist_ok =True)
tracking_dict = defaultdict(list)



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

    def on_epoch_end(self, epoch, logs={}):
        train_cc,train_rmse = utils.plot_neuron_response(self.model,self.X_train,self.y_train)
        val_cc,val_rmse = utils.plot_neuron_response(self.model,self.X_val,self.y_val)
        plt.close('all')

        self.train_track.append(train_cc)
        self.val_track.append(val_cc)
        self.train_rmse_track.append(train_rmse)
        self.val_rmse_track.append(val_rmse)

        print('neuronal_CC: ',val_cc)
        print('val_rmse: ',val_rmse)

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


for params in combination:
    run_name = "run_{:03d}".format(run_num)
    save_dir =os.path.join(experiment_directory,run_name)


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    
    model_args = {}
    data = [train_data[0],train_data[1], val_data[0], val_data[1],test_data[0],test_data[1]]


    for ind,values in enumerate(params):
        model_args[params_key[ind]] = values
        tracking_dict[params_key[ind]].append(values)




    callbacks = []
    monitor = 'val_loss'
    learning_rate_decay_factor = 0.5  # TODO use `0.1` instead (default)?

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
        patience=50,  # use `0` instead (default)?
        verbose=2,  # {0 (quiet?, default), 1 (update messages?)}
        # mode='auto',
        # baseline=None,
        restore_best_weights=True,
    )
    callbacks.append(callback)

    running_track = {'train_neu_cc':[],
                     'val_neu_cc':[],
                     'train_rmse':[],
                     'val_rmse':[]}

    neuronal_cc_callback = neuronal_CC_Callback(train_data[0],train_data[1],val_data[0],val_data[1],running_track)
    callbacks.append(neuronal_cc_callback)


    modular_network = custom_model.img_transformation_network(model,forward_inp,train_y.shape[1],model_args)
    # modular_network = custom_model.avg_downsample_network(model,forward_inp,train_y.shape[1],l2_reg=mod_l2_reg,red_dim = 32)


    modular_network.compile(loss='poisson', optimizer=optimizers.Adam(0.002), metrics=[custom_metrics.cc_met,custom_metrics.rmse_met, custom_metrics.fev_met])

    history = modular_network.fit(x=train_x, y=train_y, validation_data=(val_x, val_y),  batch_size=32, epochs=1000, verbose = 2,callbacks=callbacks)

    results = modular_network.evaluate(x=test_x,y= test_y, batch_size=test_y.shape[0], return_dict = True) #results does not give the average of the batches? Should use the full batch
    
    running_track_lst = [running_track['train_neu_cc'],running_track['val_neu_cc'],None,None,running_track['train_rmse'],running_track['val_rmse']]
    # train_avg_cc,test_avg_cc,train_avg_rmse,test_avg_rmse = utils.plot_CCs(modular_network, data,running_track_lst,save_dir,None)
    train_avg_cc,val_avg_cc,test_avg_cc,train_avg_rmse,val_avg_rmse,test_avg_rmse,test_nc_cc = utils.plot_CCs(modular_network, data,running_track_lst,save_dir,None)

    utils.plot_metrics(history,results,save_dir)

    for metric,value in results.items():
        tracking_dict[metric].append(value)


    tracking_dict['train_neuronal_rmse'].append(train_avg_rmse)
    tracking_dict['val_neuronal_rmse'].append(val_avg_rmse)
    tracking_dict['test_neuronal_rmse'].append(test_avg_rmse)
    tracking_dict['train_neuronal_cc'].append(train_avg_cc)
    tracking_dict['val_neuronal_cc'].append(val_avg_cc)
    tracking_dict['test_neuronal_cc'].append(test_avg_cc)
    tracking_dict['test_nc_cc'].append(test_nc_cc)


    df = pd.DataFrame(tracking_dict)
    df.to_csv(os.path.join(save_dir,'track.csv'))

    print(results)
    # print(running_track)
    run_num+=1

    modular_network.save_weights(os.path.join(save_dir,'my_model_checkpoint'),save_format='tf')










