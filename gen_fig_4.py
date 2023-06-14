import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import os
import utils,config
import glob
from scipy.stats import bootstrap
from scipy import stats
from scipy.stats import ttest_rel,wilcoxon, shapiro



print('starting analysis')
final_fig_dir = r'F:\Dataset_public\figures\figure_4'
spikes_root = r'F:\Dataset_public\spikes_data\validation_spikes.npy'

wn_files, orig_1, orig_2, act_32, act_64, avg_32, avg_64 ,sta_files,flash_file = config.get_file_numbers_validation()
all_spikes = np.load(spikes_root)


bin_sum=40
wn_files_resp, orig_1_resp, orig_2_resp, act_32_resp, act_64_resp, avg_32_resp,avg_64_resp = utils.sum_response_val(all_spikes, bin_sum, (wn_files, orig_1, orig_2, act_32, act_64, avg_32, avg_64),num_unique_image=100)


# #primary stability check
neuron_pass = utils.stability_check(wn_files_resp,wn_files,stability_thresh=0.3,num_unique_image=100)

orig_1_avg = utils.avg_response_repeats(orig_1_resp,orig_1,num_unique_image=100)
orig_2_avg = utils.avg_response_repeats(orig_2_resp,orig_2,num_unique_image=100)
act_32_avg = utils.avg_response_repeats(act_32_resp,act_32,num_unique_image=100)
avg_32_avg = utils.avg_response_repeats(avg_32_resp,avg_32,num_unique_image=100)


_, _, all_orig = utils.neuronal_CC_paired_stim(orig_1_avg,orig_2_avg,neuron_pass=neuron_pass,to_plot = False,xlab='orig1',ylab='orig2',save_dir=final_fig_dir)
_, _, all_act_32 = utils.neuronal_CC_paired_stim(orig_1_avg,act_32_avg,neuron_pass=neuron_pass,to_plot = False,xlab='orig1',ylab='act_32_avg',save_dir=final_fig_dir)
_, _, all_avg_32 = utils.neuronal_CC_paired_stim(orig_1_avg,avg_32_avg,neuron_pass=neuron_pass,to_plot = False,xlab='orig1',ylab='avg_32_avg',save_dir=final_fig_dir)


all_orig = np.array(all_orig)
all_act_32 = np.array(all_act_32)
all_avg_32 = np.array(all_avg_32)


cc,p1 = wilcoxon(all_orig,all_act_32)
cc,p3 = wilcoxon(all_orig,all_avg_32)
cc,p5 = wilcoxon(all_act_32,all_avg_32)


diff_results = pd.DataFrame()

diff_results['OG - all_act_32'] = np.array(all_orig)-np.array(all_act_32)
diff_results['OG - all_avg_32'] = np.array(all_orig)-np.array(all_avg_32)
diff_results['all_act_32 - all_avg_32'] = np.array(all_act_32)-np.array(all_avg_32)


# save_dir = r'C:\Users\Administrator\Desktop\cnn_analysis\cnn_analysis_multirun'
if not os.path.isdir(os.path.join(final_fig_dir,'validation_box_dist_plot')):
	os.mkdir(os.path.join(final_fig_dir,'validation_box_dist_plot'))

import seaborn as sns
fig = plt.figure()
sns.boxplot(diff_results[['OG - all_act_32','OG - all_avg_32','all_act_32 - all_avg_32']],
	flierprops={"marker": "$\circ$"})
plt.axhline(y=0.0, linestyle='--')
plt.ylim([-0.1,0.25])
plt.savefig(os.path.join(final_fig_dir,'validation_box_dist_plot','validation_boxplot_32_%s_%s_%s.jpeg'%(p1,p3,p5)))

ext='jpeg'
WIDTH_SIZE=5
HEIGHT_SIZE=9
fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
sns.boxplot(diff_results['OG - all_act_32'],
	flierprops={"marker": "$\circ$"})
plt.ylim([-0.1,0.25])
plt.axhline(y=0,ls='--')
save_dir=final_fig_dir
if save_dir is not None:
	plt.savefig(os.path.join(save_dir,'validation_box_dist_plot','boxplot32_fwd_act.%s'%(ext)))

fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
sns.boxplot(diff_results['OG - all_avg_32'],
	flierprops={"marker": "$\circ$"})
plt.ylim([-0.1,0.25])
plt.axhline(y=0,ls='--')
if save_dir is not None:
	plt.savefig(os.path.join(save_dir,'validation_box_dist_plot','boxplot32_fwd_avg.%s'%(ext)))

fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
sns.boxplot(diff_results['all_act_32 - all_avg_32'],
	flierprops={"marker": "$\circ$"})
plt.ylim([-0.1,0.25])
plt.axhline(y=0,ls='--')
if save_dir is not None:
	plt.savefig(os.path.join(save_dir,'validation_box_dist_plot','boxplot32_act_avg.%s'%(ext)))
	



results = pd.DataFrame()
results['all_orig'] = all_orig
results['all_act_32'] = all_act_32
results['all_avg_32'] = all_avg_32



utils.validation_generate_dist_plot(results,'all_orig','all_act_32',final_fig_dir)
utils.validation_generate_dist_plot(results,'all_orig','all_avg_32',final_fig_dir)
utils.validation_generate_dist_plot(results,'all_act_32','all_avg_32',final_fig_dir)

diff_results['all_act_32 - all_avg_32'] = np.array(all_act_32)-np.array(all_avg_32)

perc_inc_32 = np.divide((np.array(all_act_32) - np.array(all_avg_32)),np.array(all_avg_32))

