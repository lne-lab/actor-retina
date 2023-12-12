import os
import numpy as np
import pandas as pd
import tensorflow as tf
import utils
import custom_metrics,custom_model
from tensorflow.keras import  layers,optimizers
from cnn_models import create_ecker_cnn_model
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel,wilcoxon, shapiro,mannwhitneyu

expt_type = 'high_res'


experiment_directory = r'F:\Retina_project\Dataset_public\models\forward_model'
dataset_folder = r'F:\Retina_project\Dataset_public\models\forward_model'

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
model_args['n_out'] = 32


experiment_directory = r'F:\Retina_project\Dataset_public\models\actor_model'
final_fig_dir = r'F:\Retina_project\Dataset_public\figures\figure_3_S1a_S1b'
os.makedirs(final_fig_dir,exist_ok=True)

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

neuron_plot=range(test_y.shape[1])

fwd_FR, true_FR = utils.in_silico_generate_scatter_plot(fwd_model,test_x,test_y,mdl_type='fwd_model',save_dir=final_fig_dir,neuron_plot=neuron_plot)
act_FR,_ = utils.in_silico_generate_scatter_plot(actor_network,test_x,test_y,mdl_type='actor_network',save_dir=final_fig_dir,neuron_plot=neuron_plot)
avg_FR,_ = utils.in_silico_generate_scatter_plot(avg_network,test_x,test_y,mdl_type='avg_network',save_dir=final_fig_dir,neuron_plot=neuron_plot)





n_out = 32
perc_increase,diff_results = utils.plot_paired_test(fwd_model,avg_network,actor_network,test_x,test_y, save_dir = final_fig_dir,nout=n_out) 

RF_data = pd.read_csv(r'F:\Retina_project\Dataset_public\STA_images\STA_highres\neuron_RF.csv')

RF_size = RF_data['area_sigma1']
RGC_type = RF_data['on_off']
diff_results['RF_size'] = RF_size
diff_results['RGC_type'] = RGC_type


# print(diff_results.columns)
_,p = mannwhitneyu(diff_results['fwd - act'][diff_results['RGC_type']==1], diff_results['fwd - act'][diff_results['RGC_type']==0])

fig = plt.figure(figsize=(10,10))
sns.boxplot(data=diff_results,x='RGC_type',y = 'fwd - act',hue=diff_results['RGC_type'])
# plt.xticks([0,1,2],['Forward model','Average model','Actor model'])
plt.ylabel('Difference in neuronal reliability of Actor to High-res model (High-res - Actor)')
# plt.title('Mean firing of neurons for different models')
plt.ylim([-0.15,0.1])
plt.tight_layout()
fig.savefig(os.path.join(final_fig_dir,f'S1a.svg'))
# source_data = pd.DataFrame({'on':diff_results['fwd - act'][diff_results['RGC_type']==1],"off":diff_results['fwd - act'][diff_results['RGC_type']==0]})
# source_data.to_csv(os.path.join(final_fig_dir,f'performance_by_type_{p}.csv'))

from scipy.stats import pearsonr
# print(diff_results['fwd - act'])
# print('********')
# print(diff_results['RF_size'])

corr,p = pearsonr(diff_results['RF_size'],diff_results['fwd - act'])
fig = plt.figure(figsize=(10,10))
sns.scatterplot(data=diff_results,x='RF_size',y = 'fwd - act',hue=diff_results['RGC_type'])
# plt.xticks([0,1,2],['Forward model','Average model','Actor model'])
# plt.ylabel('Performance of neurons')
plt.ylabel('Difference in neuronal reliability of Actor to High-res model (High-res - Actor)')
plt.xlabel('RF size')
plt.ylim([-0.15,0.1])
plt.xlim([0.00,0.045])
# plt.title('Mean firing of neurons for different models')
plt.tight_layout()
fig.savefig(os.path.join(final_fig_dir,f'S1b.svg'))
# source_data = pd.DataFrame({'RF_size':diff_results['RF_size'],"performance":diff_results['fwd - act']})
# source_data.to_csv(os.path.join(final_fig_dir,f'performance_by_size_corr_{corr}_p_{p}.csv'))