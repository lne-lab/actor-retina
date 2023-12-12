import numpy as np
import os
import glob
from scipy import stats
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel,wilcoxon, shapiro
from matplotlib import rcParams
from tensorflow.keras import datasets, layers, models,regularizers,optimizers
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MaxNLocator





def sum_response(all_spikes, n_bins, file_numbers, num_unique_image = 200):
	#sum up the response within the window
	#all_spikes: binned spike data with shape n_files, n_neurons x bins. bins is 10ms 
	#nbins: how many bins to sum up the data.
	#file_numbers: list corresponding file numbers for WN, Nat, Paired

	stim_change = np.arange(0,all_spikes.shape[2], all_spikes.shape[2]//num_unique_image) #return the index where a stimulus change occurs

	summed_response=np.empty((all_spikes.shape[0],all_spikes.shape[1],len(stim_change))) 
	for ind, i in enumerate(stim_change):
		if all_spikes.shape[1]>1:
			summed_response[:,:,ind] = np.squeeze(np.sum(all_spikes[:,:,i:i+n_bins], axis = 2))
		else:
			summed_response[:,:,ind] = np.sum(all_spikes[:,:,i:i+n_bins], axis = 2)

	wn_file,nat_file = file_numbers
	wn_resp,nat_resp = None, None

	for file in range(summed_response.shape[0]):
		resp = summed_response[file,:,:]
		if file in wn_file:
			if wn_resp is None:
				wn_resp = resp
			else:
				wn_resp = np.concatenate([wn_resp,resp],axis=1)
		elif file in nat_file:
			if nat_resp is None:
				nat_resp = resp
			else:
				nat_resp = np.concatenate([nat_resp,resp],axis=1)

	return wn_resp.T,nat_resp.T

def sum_response_5res(all_spikes, n_bins, file_numbers, num_unique_image = 500):
	#sum up the response within the window
	#all_spikes: binned spike data with shape n_files, n_neurons x bins. bins is 10ms 
	#nbins: how many bins to sum up the data.
	#file_numbers: list corresponding file numbers for WN, Nat, Paired

	stim_change = np.arange(0,all_spikes.shape[2], all_spikes.shape[2]//num_unique_image) #return the index where a stimulus change occurs
	#each stimulation is presented for 200 ms followed by 400ms of grey screen. thus, every 600ms there will be a change in stimulus

	summed_response=np.empty((all_spikes.shape[0],all_spikes.shape[1],len(stim_change))) 
	for ind, i in enumerate(stim_change):
		summed_response[:,:,ind] = np.squeeze(np.sum(all_spikes[:,:,i:i+n_bins], axis = 2))

	wn_files, files_512, files_256, files_128, files_64, files_32 = file_numbers
	# wn_resp,nat_resp,paired_resp = None, None, None

	wn_files_resp, files_512_resp, files_256_resp, files_128_resp, files_64_resp, files_32_resp = None, None, None, None, None, None,

	for file in range(summed_response.shape[0]):
		resp = summed_response[file,:,:]

		if file in wn_files:
			if wn_files_resp is None:
				wn_files_resp = resp
			else:
				wn_files_resp = np.concatenate([wn_files_resp,resp],axis=1)

		elif file in files_512:
			if files_512_resp is None:
				files_512_resp = resp
			else:
				files_512_resp = np.concatenate([files_512_resp,resp],axis=1)

		elif file in files_256:
			if files_256_resp is None:
				files_256_resp = resp
			else:
				files_256_resp = np.concatenate([files_256_resp,resp],axis=1)

		elif file in files_128:
			if files_128_resp is None:
				files_128_resp = resp
			else:
				files_128_resp = np.concatenate([files_128_resp,resp],axis=1)

		elif file in files_64:
			if files_64_resp is None:
				files_64_resp = resp
			else:
				files_64_resp = np.concatenate([files_64_resp,resp],axis=1)

		elif file in files_32:
			if files_32_resp is None:
				files_32_resp = resp
			else:
				files_32_resp = np.concatenate([files_32_resp,resp],axis=1)



	return wn_files_resp.T, files_512_resp.T, files_256_resp.T, files_128_resp.T, files_64_resp.T, files_32_resp.T

def sum_response_val(all_spikes, n_bins, file_numbers, num_unique_image = 500):
	#sum up the response within the window
	#all_spikes: binned spike data with shape n_files, n_neurons x bins. bins is 10ms 
	#nbins: how many bins to sum up the data.
	#file_numbers: list corresponding file numbers for WN, Nat, Paired

	stim_change = np.arange(0,all_spikes.shape[2], all_spikes.shape[2]//num_unique_image) #return the index where a stimulus change occurs
	#each stimulation is presented for 200 ms followed by 400ms of grey screen. thus, every 600ms there will be a change in stimulus

	summed_response=np.empty((all_spikes.shape[0],all_spikes.shape[1],len(stim_change))) 
	for ind, i in enumerate(stim_change):
		if all_spikes.shape[1]>1:
			summed_response[:,:,ind] = np.squeeze(np.sum(all_spikes[:,:,i:i+n_bins], axis = 2))
		else:
			summed_response[:,:,ind] = np.sum(all_spikes[:,:,i:i+n_bins], axis = 2)

		# summed_response[:,:,ind] = np.squeeze(np.sum(all_spikes[:,:,i:i+n_bins], axis = 2))

	wn_files, orig_1_files, orig_2_files, act_32_files, act_64_files, avg_32_files, avg_64_files = file_numbers
	# wn_resp,nat_resp,paired_resp = None, None, None

	wn_files_resp, orig_1_resp, orig_2_resp, act_32_resp, act_64_resp, avg_32_resp, avg_64_resp = None, None, None, None, None, None,None

	for file in range(summed_response.shape[0]):
		resp = summed_response[file,:,:]

		if file in wn_files:
			if wn_files_resp is None:
				wn_files_resp = resp
			else:
				wn_files_resp = np.concatenate([wn_files_resp,resp],axis=1)

		elif file in orig_1_files:
			if orig_1_resp is None:
				orig_1_resp = resp
			else:
				orig_1_resp = np.concatenate([orig_1_resp,resp],axis=1)

		elif file in orig_2_files:
			if orig_2_resp is None:
				orig_2_resp = resp
			else:
				orig_2_resp = np.concatenate([orig_2_resp,resp],axis=1)

		elif file in act_32_files:
			if act_32_resp is None:
				act_32_resp = resp
			else:
				act_32_resp = np.concatenate([act_32_resp,resp],axis=1)

		elif file in act_64_files:
			if act_64_resp is None:
				act_64_resp = resp
			else:
				act_64_resp = np.concatenate([act_64_resp,resp],axis=1)

		elif file in avg_32_files:
			if avg_32_resp is None:
				avg_32_resp = resp
			else:
				avg_32_resp = np.concatenate([avg_32_resp,resp],axis=1)
		elif file in avg_64_files:
			if avg_64_resp is None:
				avg_64_resp = resp
			else:
				avg_64_resp = np.concatenate([avg_64_resp,resp],axis=1)


	return wn_files_resp.T, orig_1_resp.T, orig_2_resp.T, act_32_resp.T, act_64_resp.T, avg_32_resp.T,avg_64_resp.T



def get_stim(stim_path,file_numbers,num_unique_image=500,wn_res = 64, nat_res=128,num_images=30000,to_norm = True):
	#load the stimulus files and segment them by files
	#return stim: (n_stims x res x res) 
	wn_file,nat_file = file_numbers
	num_stim_files = len(glob.glob(os.path.join(stim_path,'*.npy')))
	stim_change = np.arange(0,num_images, num_images//num_unique_image)
	wn_stim,nat_stim,paired_stim = None,None,None
	for file in range(num_stim_files):
		stim = np.load(os.path.join(stim_path,'%s.npy'%(file+1))) 
		stim = stim[stim_change,:,:]
		if to_norm:
			assert(np.max(stim>1))
			stim=stim/255

		if file in wn_file:
			if wn_stim is None:
				wn_stim = stim
			else:
				wn_stim = np.concatenate([wn_stim,stim],axis=0)
		elif file in nat_file:
			if nat_stim is None:
				nat_stim = stim
			else:
				nat_stim = np.concatenate([nat_stim,stim],axis=0)
	return wn_stim,nat_stim


def avg_response_repeats(spikes,files,num_unique_image = 500, num_reps = 10):
	#averages the response across the repeats to the same stimulus
	all_neuron_response = np.empty((len(files)*num_unique_image//num_reps,spikes.shape[1]))
	for neuron in range(spikes.shape[1]):
		neuron_response = []
		for file_ind in range(len(files)):
			tmp = (file_ind)*num_unique_image
			responses = spikes[tmp:tmp+num_unique_image, neuron]
			responses = np.mean(responses.reshape((num_reps,num_unique_image//num_reps)),axis = 0)
			neuron_response.extend(responses)
		all_neuron_response[:,neuron] = neuron_response

	return all_neuron_response
def avg_response_odd_even(spikes,files,num_unique_image = 500, num_reps = 10):
	#averages the response across the repeats to the same stimulus
	all_neuron_response_odd = np.empty((len(files)*num_unique_image//num_reps,spikes.shape[1]))
	all_neuron_response_even = np.empty((len(files)*num_unique_image//num_reps,spikes.shape[1]))

	odd_ind = np.arange(1,num_reps,2)
	even_ind = np.arange(0,num_reps,2)

	for neuron in range(spikes.shape[1]):
		neuron_response_odd = []
		neuron_response_even = []
		for file_ind in range(len(files)):
			tmp = (file_ind)*num_unique_image
			responses = spikes[tmp:tmp+num_unique_image, neuron]
			responses = responses.reshape((num_reps,num_unique_image//num_reps))

			responses_odd = np.mean(responses[odd_ind,:],axis = 0)
			responses_even = np.mean(responses[even_ind,:],axis = 0)

			neuron_response_odd.extend(responses_odd)
			neuron_response_even.extend(responses_even)


		all_neuron_response_odd[:,neuron] = neuron_response_odd
		all_neuron_response_even[:,neuron] = neuron_response_even

	return all_neuron_response_odd,all_neuron_response_even


def avg_nat_stim(nat_stim,nat_file,num_unique_image=500,num_reps = 10):
	#average the 10 repeats of the image to give the same size format as response
	#Essentially reducing 10 identical image to a single image
	all_images = np.empty((len(nat_file)*num_unique_image//num_reps,nat_stim.shape[1],nat_stim.shape[2]))
	for file_ind in range(len(nat_file)):
		tmp = (file_ind)*num_unique_image
		tmp2 = (file_ind)*(num_unique_image//num_reps)
		all_images[tmp2:tmp2+(num_unique_image//num_reps),:,:] = nat_stim[tmp:tmp+(num_unique_image//num_reps),:,:]
	return all_images

def stability_check(wn_spikes,wn_file,stability_thresh=0.3,num_unique_image=500, num_reps = 10):
	#perform stability check of neuronal response to the same WN stimulus over the course of experiment
	neuron_pass = []
	for neuron in range(wn_spikes.shape[1]):
		neuron_response = np.empty((num_unique_image//num_reps,len(wn_file)))
		for file_ind, file in enumerate(wn_file):
			responses = np.empty((wn_spikes.shape[1], num_unique_image))
			tmp = (file_ind)*num_unique_image
			responses = wn_spikes[tmp:tmp+num_unique_image,neuron]
			responses = np.mean(responses.reshape((num_reps,num_unique_image//num_reps)),axis=0)
			neuron_response[:,file_ind] = responses[:]
		neuron_response = neuron_response.T
		CC = np.corrcoef(neuron_response)
		il1 = np.tril_indices(CC.shape[0],k=-1) #its a symmetrical matrix, so we just take lower triangle and average the correlation between the WN stimulus
		stability_mean = np.nanmean(CC[il1])
		if stability_mean>stability_thresh:
			# print(stability_mean)
			neuron_pass.append(neuron)
	return neuron_pass

def train_val_test_split_odd_even(nat_stim, nat_resp_odd,nat_resp_even,val_size,test_size,neuron_pass = None):
	num_sample = nat_stim.shape[0]
	index = np.arange(num_sample)
	np.random.shuffle(index)
	train_ind, validate_ind, test_ind = np.split(index, [int((1-val_size-test_size)*len(index)), int((1-test_size)*len(index))])

	if neuron_pass is not None:
		nat_resp_even = nat_resp_even[:,neuron_pass]
		nat_resp_odd = nat_resp_odd[:,neuron_pass]
	
	stim_train = nat_stim[train_ind,:,:]
	stim_val = nat_stim[validate_ind,:,:]
	stim_test = nat_stim[test_ind,:,:]

	nat_train_odd = nat_resp_odd[train_ind,:]
	nat_val_odd = nat_resp_odd[validate_ind,:]
	nat_test_odd = nat_resp_odd[test_ind,:]

	nat_train_even = nat_resp_even[train_ind,:]
	nat_val_even = nat_resp_even[validate_ind,:]
	nat_test_even = nat_resp_even[test_ind,:]


	return stim_train,stim_val,stim_test,nat_train_odd,nat_val_odd,nat_test_odd,nat_train_even,nat_val_even,nat_test_even

def save_train_val_test(save_dir, train_x,val_x,test_x,train_y,val_y,test_y):
	os.makedirs(save_dir,exist_ok =True)
	np.save(os.path.join(save_dir,'train_x.npy'),train_x)
	np.save(os.path.join(save_dir,'val_x.npy'),val_x)
	np.save(os.path.join(save_dir,'test_x.npy'),test_x)
	np.save(os.path.join(save_dir,'train_y.npy'),train_y)
	np.save(os.path.join(save_dir,'val_y.npy'),val_y)
	np.save(os.path.join(save_dir,'test_y.npy'),test_y)

def load_train_val_test(load_dir):
	train_x = np.load(os.path.join(load_dir,'train_x.npy'))
	val_x = np.load(os.path.join(load_dir,'val_x.npy'))
	test_x = np.load(os.path.join(load_dir,'test_x.npy'))
	train_y = np.load(os.path.join(load_dir,'train_y.npy'))
	val_y = np.load(os.path.join(load_dir,'val_y.npy'))
	test_y = np.load(os.path.join(load_dir,'test_y.npy'))
	return train_x,val_x,test_x,train_y,val_y,test_y

def split_test_y(test_y):
	test_y_odd = test_y[:test_y.shape[0]//2,:]
	test_y_even = test_y[(test_y.shape[0]//2):,:]
	return test_y_odd,test_y_even

def merge_test_y(test_y):
	test_y_odd,test_y_even = split_test_y(test_y)
	test_y_merged = np.mean(np.stack([test_y_odd,test_y_even]),axis=0)
	return test_y_merged


def corrcoef(x, y):
    """Return Pearson product-moment correlations coefficients.

    This is a wrapper around `np.corrcoef` to avoid:
        `RuntimeWarning: invalid value encountered in true_divide`.
    """

    assert len(x) > 0, len(x)
    assert len(y) > 0, len(y)
    assert len(x) == len(y), (len(x), len(y))

    is_x_deterministic = np.all(x == x[0])  # i.e. array filled with a unique value
    is_y_deterministic = np.all(y == y[0])  # i.e. array filled with a unique value
    if is_x_deterministic and is_y_deterministic:
        r = 1.0
    elif is_x_deterministic or is_y_deterministic:
        r = 0.0
    else:
        r = np.corrcoef(x, y)[0, 1]

    return r


def deepupdate(dict_1, dict_2):

    for key, value in dict_2.items():
        if key not in dict_1:
            dict_1[key] = value
        else:
            if not isinstance(dict_1[key], dict):
                dict_1[key] = value
            else:
                dict_1[key] = deepupdate(dict_1[key], value)

    return dict_1


def plot_neuron_response(mdl,X,y,visual_type='test',n_images=200,save_dir = None):

	all_ind = np.arange(0,n_images,1)
	prediction = mdl(X[:min(n_images,y.shape[0]),:,:]) #(n test by n neurons)

	neuron_all_cc = []
	neuron_all_rmse = [] 

	for n in range(y.shape[1]):
		neuron_response = prediction[:len(all_ind),n]
		true_response = y[:len(all_ind),n]

		
		# cc,p = stats.pearsonr(neuron_response,true_response)
		cc = corrcoef(neuron_response,true_response)
		neuron_all_cc.append(cc)
		rmse = np.sqrt(np.nanmean(np.square(neuron_response - true_response)))
		neuron_all_rmse.append(rmse)

		if save_dir is not None:
			fig = plt.figure()
			plt.scatter(neuron_response,true_response)
			max_axis = max(np.max(neuron_response),np.max(true_response))
			min_axis = min(np.min(neuron_response),np.min(true_response))
			plt.xlim(0,max_axis)
			plt.ylim(0,max_axis)

			if visual_type == 'test':
				if not os.path.isdir(os.path.join(save_dir,'neuron_scatter_test')):
					os.mkdir(os.path.join(save_dir,'neuron_scatter_test'))
				fig.savefig(os.path.join(save_dir,'neuron_scatter_test','neuron_%d.jpeg'%(n)))
			if visual_type == 'train':
				if not os.path.isdir(os.path.join(save_dir,'neuron_scatter_train')):
					os.mkdir(os.path.join(save_dir,'neuron_scatter_train'))
				fig.savefig(os.path.join(save_dir,'neuron_scatter_train','neuron_%d.jpeg'%(n)))

			plt.close('all')
	return np.nanmean(neuron_all_cc),np.nanmean(neuron_all_rmse)

def plot_metrics(history, results=None,save_dir = None,custom_metrics = False):
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

	if not custom_metrics:
		metrics = history.history.keys()
		metric_dict = history.history
	else:
		metrics = history.keys()
		metric_dict = history

	for ind,metric in enumerate(metrics):
		if 'val' in metric:
			continue
		if save_dir != None:
			fig = plt.figure()
			plt.plot(metric_dict[metric],label = 'train')
			if  metric != 'lr':
				plt.plot(metric_dict['val_'+metric],label = 'validation')
				if results != None:
					plt.axhline(results[metric],linestyle ='--',label='test')
			plt.title(metric)
			plt.legend()
		
			fig.savefig(os.path.join(save_dir, metric.replace("/", "_") + '.jpeg'))
			plt.close('all')

def plot_CCs(mdl,data,running_track,result_save_root,run_num):
	X_train,y_train,X_val,y_val,X_test,y_test = data


	train_neuronal_cc_runningtrack,val_neuronal_cc_runningtrack,train_nccc_runningtrack,val_nccc_runningtrack,train_neuronal_rmse_runningtrack,val_neuronal_rmse_runningtrack = running_track
	if run_num is not None:
		save_dir = os.path.join(result_save_root,'run_%d'%(run_num))
	else:
		save_dir = result_save_root

	train_avg_cc,train_avg_rmse = plot_neuron_response(mdl,X_train,y_train,visual_type='train',save_dir=save_dir)
	val_avg_cc,val_avg_rmse = plot_neuron_response(mdl,X_val,y_val,visual_type='val',save_dir=save_dir)
	test_avg_cc,test_avg_rmse = plot_neuron_response(mdl,merge_test_y(X_test),merge_test_y(y_test),visual_type='test',save_dir=save_dir)
	neuronal_cc_train_val = {'neuronal_cc':train_neuronal_cc_runningtrack,'val_neuronal_cc':val_neuronal_cc_runningtrack}
	neuronal_cc_test = {'neuronal_cc':test_avg_cc}
	plot_metrics(neuronal_cc_train_val,neuronal_cc_test,custom_metrics=True,save_dir =save_dir)

	neuronal_rmse_train_val = {'neuronal_rmse':train_neuronal_rmse_runningtrack,'val_neuronal_rmse':val_neuronal_rmse_runningtrack}
	neuronal_rmse_test = {'neuronal_rmse':test_avg_rmse}
	plot_metrics(neuronal_rmse_train_val,neuronal_rmse_test,custom_metrics=True,save_dir = save_dir)

	#plot ensemble CC
	plot_ensemble_response(mdl,X_train,y_train,visual_type='train',save_dir=save_dir)
	plot_ensemble_response(mdl,X_test,y_test,visual_type='test',save_dir=save_dir)
	test_nc_cc = noise_corrected_cc(mdl,X_test,y_test)

	return train_avg_cc,val_avg_cc,test_avg_cc,train_avg_rmse,val_avg_rmse,test_avg_rmse,test_nc_cc
def plot_ensemble_response(mdl,X,y,visual_type='test',n_images=50,save_dir = None):
	prediction = mdl(X[:n_images,:,:])
	ensemble_all_cc = []
	for n in range(n_images):
		neuron_response = prediction[n,:]
		true_response = y[n,:]


		if save_dir is not None:
			fig = plt.figure()
			plt.scatter(neuron_response,true_response)
			max_axis = max(np.max(neuron_response),np.max(true_response))
			min_axis = min(np.min(neuron_response),np.min(true_response))
			plt.xlim(0,max_axis)
			plt.ylim(0,max_axis)

			if visual_type == 'test':
				if not os.path.isdir(os.path.join(save_dir,'ensemble_scatter_test')):
					os.mkdir(os.path.join(save_dir,'ensemble_scatter_test'))
				fig.savefig(os.path.join(save_dir,'ensemble_scatter_test','image_%d.jpeg'%(n)))
			if visual_type == 'train':
				if not os.path.isdir(os.path.join(save_dir,'ensemble_scatter_train')):
					os.mkdir(os.path.join(save_dir,'ensemble_scatter_train'))
				fig.savefig(os.path.join(save_dir,'ensemble_scatter_train','image_%d.jpeg'%(n)))
		plt.close('all')

def noise_corrected_cc(mdl,X,y,n_images = 200, type = 'eval'):
	neuron_all_cc = []
	y_odd,y_even = split_test_y(y)
	stim_odd, stim_even = split_test_y(X)
	prediction = mdl(stim_odd) #stim odd and stim even is the same



	for n in range(y.shape[1]):
		neuron_response = prediction[:,n]
		true_response_odd = y_odd[:,n]
		true_response_even = y_even[:,n]

		#Check this part
		r_mdl_o = corrcoef(neuron_response,true_response_odd)
		r_mdl_e = corrcoef(neuron_response,true_response_even)
		r_oe = corrcoef(true_response_odd,true_response_even)

		nc_cc = (0.5*(r_mdl_o+r_mdl_e))/math.sqrt(r_oe)
		
		neuron_all_cc.append(nc_cc)

	return np.mean(neuron_all_cc)

def neuronal_CC_paired_stim(nat_resp,paired_resp,neuron_pass=None, to_plot = False,xlab=None,ylab=None,save_dir=None):
	all_neuronal_CC = []
	if neuron_pass is not None:
		assert len(neuron_pass) > 2
	if neuron_pass is not None:
		nat_resp = nat_resp[:,neuron_pass]
		paired_resp = paired_resp[:,neuron_pass]


	for n in range(nat_resp.shape[1]):
		response = nat_resp[:,n]
		paired_response = paired_resp[:,n]
		R2 = corrcoef(response,paired_response)**2
		all_neuronal_CC.append(R2)

	if to_plot:
		os.makedirs(os.path.join(save_dir,'validation_neuron_scatter'),exist_ok=True)
		for n in range(nat_resp.shape[1]):
			fig = plt.figure(figsize = (5,5))
			response = nat_resp[:,n]
			paired_response = paired_resp[:,n]
			R2 = corrcoef(response,paired_response)**2
			data_csv = pd.DataFrame()
			data_csv['x'] = paired_response
			data_csv['y'] = response

			sns.scatterplot(x=paired_response,y=response)
			sns.despine(offset=20)
			max_axis = max(np.max(response),np.max(paired_response))
			min_axis = min(np.min(response),np.min(paired_response))

			# plt.plot([0,max_axis],[0,max_axis], 'k-', alpha=0.75, zorder=0)
			plt.xlim(-1,max_axis+int(0.1*max_axis))
			plt.xlabel(ylab)#swabbing the axis
			plt.ylim(-1,max_axis+int(0.1*max_axis))
			plt.ylabel(xlab)#swapping the axis label
			plt.title('R2: %f'%(R2))
			plt.tight_layout()
			
			os.makedirs(os.path.join(save_dir,'validation_neuron_scatter','neuron_%d'%(n)),exist_ok=True)
			plt.savefig(os.path.join(save_dir,'validation_neuron_scatter','neuron_%d'%(n),'%s_%s_scatter_R2_%0.3f.svg'%(xlab,ylab,R2)))
			data_csv.to_csv(os.path.join(save_dir,'validation_neuron_scatter','neuron_%d'%(n),'%s_%s_scatter_R2_%0.3f.csv'%(xlab,ylab,R2)))
			# plt.show()

	return np.nanmean(all_neuronal_CC),np.nanstd(all_neuronal_CC),all_neuronal_CC

def validation_generate_dist_plot(results,first_condition,second_condition,final_fig_dir,ylim=(0.5,1)):
	tmp_np = results[[first_condition,second_condition]].to_numpy()
	g = sns.catplot(aspect=5/9,data=pd.melt(results[[first_condition,second_condition]]), x="variable", y="value", jitter=False)
	plt.ylim(ylim[0], ylim[1])
	plt.title('r2')
	sns.despine(offset=10)
	for i in tmp_np:
		g.ax.plot([first_condition,second_condition],[i[0],i[1]])
	# fig = g.get_figure()
	plt.tight_layout()
	os.makedirs(os.path.join(final_fig_dir,'validation_box_dist_plot'),exist_ok=True)
	g.fig.savefig(os.path.join(final_fig_dir,'validation_box_dist_plot','%s_%s.svg'%(first_condition,second_condition)))

def load_params_actor(load_dir,model_args):
	tracking = pd.read_csv(os.path.join(load_dir,'track.csv'))
	model_args['n_channel'] = tracking.n_channel.iloc[-1]
	model_args['kernal_size']=tracking.kernal_size.iloc[-1]
	model_args['l2_reg'] = tracking.l2_reg.iloc[-1]
	model_args['n_out'] = tracking.n_out.iloc[-1]

	return model_args

def get_neuronal_cc_vector(mdl,X,y,n_images=200,save_dir = None):


	prediction = mdl(X) #(n test by n neurons)
	neuron_all_cc = []
	neuron_all_rmse = [] 

	for n in range(y.shape[1]):

		neuron_response = prediction[:n_images,n]
		true_response = y[:n_images,n]

		cc = corrcoef(neuron_response,true_response)
		neuron_all_cc.append(cc)

	return neuron_all_cc


def plot_paired_test(fwd_model,avg_model,actor_model,X,y,save_dir = None, nout=None):


	results = pd.DataFrame()
	results['fwd_neuronal_R2'] = list(map(lambda x: x ** 2,get_neuronal_cc_vector(fwd_model,X,y)))
	results['avg_neuronal_R2'] = list(map(lambda x: x ** 2,get_neuronal_cc_vector(avg_model,X,y)))
	diff_results = pd.DataFrame()
	


	if actor_model is not None:
		results['actor_neuronal_R2'] = list(map(lambda x: x ** 2,get_neuronal_cc_vector(actor_model,X,y)))
		diff_results['fwd - act'] = results['fwd_neuronal_R2'] - results['actor_neuronal_R2']
		diff_results['fwd - avg'] = results['fwd_neuronal_R2'] - results['avg_neuronal_R2']
		diff_results['act - avg'] = results['actor_neuronal_R2'] - results['avg_neuronal_R2']
		perc_increase = np.divide((results['actor_neuronal_R2'] - results['avg_neuronal_R2']),results['avg_neuronal_R2'])
		# perc_increase = (results['actor_neuronal_R2'] - results['avg_neuronal_R2'])


	if actor_model is not None:
		cc2,p2 = wilcoxon(results['fwd_neuronal_R2'],results['actor_neuronal_R2'],alternative='greater')
		cc1,p1 = wilcoxon(results['fwd_neuronal_R2'],results['avg_neuronal_R2'],alternative='greater')
		cc3,p3 = wilcoxon(results['actor_neuronal_R2'],results['avg_neuronal_R2'],alternative='greater')


	ext = 'jpeg'
	fig = plt.figure()
	sns.boxplot(diff_results,
		flierprops={"marker": "$\circ$"})
	plt.ylim([-0.1,0.15])
	plt.axhline(y=0,ls='--')
	if save_dir is not None:
		if not os.path.isdir(os.path.join(save_dir,'insilico_box_dist_plot')):
			os.mkdir(os.path.join(save_dir,'insilico_box_dist_plot'))
		plt.savefig(os.path.join(save_dir,'insilico_box_dist_plot','one_tail_boxplot_%s_act_%s_avg_%s_actavg_%s.%s'%(nout,p2,p1,p3,ext)))
		diff_results.to_csv(os.path.join(save_dir,'all_box_data.csv'))

	WIDTH_SIZE=5
	HEIGHT_SIZE=9
	fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
	sns.boxplot(diff_results['fwd - act'],
		flierprops={"marker": "$\circ$"})
	plt.ylim([-0.1,0.15])
	plt.axhline(y=0,ls='--')
	if save_dir is not None:
		if not os.path.isdir(os.path.join(save_dir,'insilico_box_dist_plot')):
			os.mkdir(os.path.join(save_dir,'insilico_box_dist_plot'))
		plt.savefig(os.path.join(save_dir,'insilico_box_dist_plot','boxplot_fwd_act.%s'%(ext)))
	
	fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
	sns.boxplot(diff_results['fwd - avg'],
		flierprops={"marker": "$\circ$"})
	plt.ylim([-0.1,0.15])
	plt.axhline(y=0,ls='--')
	if save_dir is not None:
		if not os.path.isdir(os.path.join(save_dir,'insilico_box_dist_plot')):
			os.mkdir(os.path.join(save_dir,'insilico_box_dist_plot'))
		plt.savefig(os.path.join(save_dir,'insilico_box_dist_plot','boxplot_fwd_avg.%s'%(ext)))
	
	fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
	sns.boxplot(diff_results['act - avg'],
		flierprops={"marker": "$\circ$"})
	plt.ylim([-0.1,0.15])
	plt.axhline(y=0,ls='--')
	if save_dir is not None:
		if not os.path.isdir(os.path.join(save_dir,'insilico_box_dist_plot')):
			os.mkdir(os.path.join(save_dir,'insilico_box_dist_plot'))
		plt.savefig(os.path.join(save_dir,'insilico_box_dist_plot','boxplot_act_avg.%s'%(ext)))
	


	tmp_np = results[['fwd_neuronal_R2','avg_neuronal_R2']].to_numpy()
	g = sns.catplot(aspect=5/9,data=pd.melt(results[['fwd_neuronal_R2','avg_neuronal_R2']]), x="variable", y="value", jitter=False)
	plt.ylim(-0.05, 1)
	plt.title('R2 distribution')

	sns.despine(offset=10)
	for i in tmp_np:
		g.ax.plot(['fwd_neuronal_R2','avg_neuronal_R2'],[i[0],i[1]])
	# fig = g.get_figure()
	plt.tight_layout()
	if save_dir is not None:
		g.fig.savefig(os.path.join(save_dir,'insilico_box_dist_plot','fwd_avg.%s'%(ext)))


	tmp_np = results[['fwd_neuronal_R2','actor_neuronal_R2']].to_numpy()
	g = sns.catplot(aspect=5/9,data=pd.melt(results[['fwd_neuronal_R2','actor_neuronal_R2']]), x="variable", y="value", jitter=False)
	plt.ylim(-0.05, 1)
	plt.title('R2 distribution')
	sns.despine(offset=10)
	for i in tmp_np:
		g.ax.plot(['fwd_neuronal_R2','actor_neuronal_R2'],[i[0],i[1]])
	# fig = g.get_figure()
	plt.tight_layout()
	if save_dir is not None:
		g.fig.savefig(os.path.join(save_dir,'insilico_box_dist_plot','fwd_act.%s'%(ext)))

	tmp_np = results[['actor_neuronal_R2','avg_neuronal_R2']].to_numpy()
	g = sns.catplot(aspect=5/9,data=pd.melt(results[['actor_neuronal_R2','avg_neuronal_R2']]), x="variable", y="value", jitter=False)
	plt.ylim(-0.05, 1)
	plt.title('R2 distribution')
	sns.despine(offset=10)
	for i in tmp_np:
		g.ax.plot(['actor_neuronal_R2','avg_neuronal_R2'],[i[0],i[1]])
	# fig = g.get_figure()
	plt.tight_layout()
	if save_dir is not None:
		g.fig.savefig(os.path.join(save_dir,'insilico_box_dist_plot','act_avg.%s'%(ext)))

	return perc_increase,diff_results



def in_silico_generate_scatter_plot(mdl,X,y,mdl_type=None,save_dir = None, neuron_plot=None,n_images=200):

	# all_ind = np.arange(0,min(n_images,y.shape[0]),1)
	# prediction = mdl(X[:min(n_images,y.shape[0]),:,:]) #(n test by n neurons)
	all_ind = np.arange(0,n_images,1)
	prediction = mdl(X[:n_images,:,:]) #(n test by n neurons)
	res_pred = np.empty((len(neuron_plot),n_images))
	res_true = np.empty((len(neuron_plot),n_images))


	for n in neuron_plot:
		true_response = y[:len(all_ind),n]
		neuron_response = prediction[:len(all_ind),n]
		cc = corrcoef(neuron_response,true_response)**2
		data_csv = pd.DataFrame()
		data_csv['x'] = neuron_response
		data_csv['y'] = true_response
		
		res_pred[n,:] = neuron_response 
		res_true[n,:] = true_response

		if save_dir is not None:
			# fig = plt.figure()
			fig = plt.figure(figsize = (5,5))
			# plt.scatter(neuron_response,true_response)
			sns.scatterplot(x=neuron_response,y=true_response)
			max_axis = max(np.max(neuron_response),np.max(true_response))
			min_axis = min(np.min(neuron_response),np.min(true_response))
			plt.xlim(-1,max_axis+int(0.1*max_axis))
			plt.ylim(-1,max_axis+int(0.1*max_axis))
			plt.ylabel('ground truth')
			plt.xlabel('predicted response')
			plt.title('R2: %f'%(cc))
			# sns.despine(offset=20)
			plt.tight_layout()

			os.makedirs(os.path.join(save_dir,'in_silico_neuron_scatter','neuron_%d'%(n)),exist_ok=True)
			fig.savefig(os.path.join(save_dir,'in_silico_neuron_scatter','neuron_%d'%(n),'%s_scatter_%d_R2_%0.2f.svg'%(mdl_type, n,cc)))
			fig.savefig(os.path.join(save_dir,'in_silico_neuron_scatter','neuron_%d'%(n),'%s_scatter_%d_R2_%0.2f.jpeg'%(mdl_type, n,cc)))
			data_csv.to_csv(os.path.join(save_dir,'in_silico_neuron_scatter','neuron_%d'%(n),'%s_scatter_%d_R2_%0.2f.csv'%(mdl_type, n,cc)))
	
	# fig,ax = plt.figure(figsize=(10,10))
	# plt.imshow(res)
	# plt.colorbar()
	# plt.tight_layout()
	# fig.savefig(os.path.join(save_dir,'in_silico_neuron_scatter','%s_FRdiff_heatmap.svg'%(mdl_type)))
	# fig.savefig(os.path.join(save_dir,'in_silico_neuron_scatter','%s_FRdiff_heatmap.jpeg'%(mdl_type)))



	return res_pred,res_true



def visualize_transformation(model,inp,layer_track,sub_model= None,image_save=None,n_image=5,all_ells=None,transform_type = 'actor'):
	#https://taiolifrancesco.medium.com/what-a-cnn-see-visualizing-intermediate-output-of-the-conv-layers-with-tensorflow-f935f55f9e8d

	if not os.path.exists(image_save):
		os.mkdir('visualize_transformation')


	all_image = []
	all_orig = []
	for j in range(n_image):
		print("Processing image {}...".format(str(j)))
		stim_image = inp[[j],:,:]
		image_dir = os.path.join(image_save,'visualize_transformation_{}'.format(transform_type),'image_%d'%(j))

		if not os.path.isdir(image_dir):
			os.makedirs(image_dir)


		if sub_model != None:
			compare_fig, compare_ax = fig, ax = plt.subplots(nrows=1, ncols=2)
			compare_ax[0].imshow(np.squeeze(stim_image[:,:]))
			# plot_spatial_w_all_ellipse(None,all_ells,None,None,compare_ax[0])

		fig = plt.figure()
		plt.imshow(np.squeeze(stim_image[:,:]))
		plt.colorbar()
		all_orig.append(stim_image[:,:])

		if image_save != None:
			fig.savefig(os.path.join(image_dir,  'stim.jpeg'))
		stim_image = np.expand_dims(stim_image, axis=-1)

		# print(model.layers)
		layer_outputs = [layer.output for layer in model.layers]
		activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
		activations = activation_model.predict(stim_image)

		conv_indixes = []


		for i in range(len(activations)):
			if any([track in model.layers[i].name for track in layer_track]):
				conv_indixes.append(i)

		print(conv_indixes)
		for i, conv_ix in enumerate(conv_indixes):
			print(i,conv_ix,model.layers[conv_ix].name)
			up_layer = False
			if conv_ix == conv_indixes[-1]:
				up_layer = True
			image = plot_layer(model.layers[conv_ix].name, activations[conv_ix],image_save=image_save,image_dir=image_dir,up_layer=up_layer)


			if all_ells!=None and activations[conv_ix].shape[3] == 1:
				compare_ax[1].imshow(np.squeeze(activations[conv_ix][:,:]))
				plot_spatial_w_all_ellipse(None,all_ells,None,None,compare_ax[1])
				compare_fig.savefig(os.path.join(image_dir,  'stim_transformed.jpeg'))

		plt.close('all')
		all_image.append(image)
	return all_image,all_orig

def plot_layer(name, activation, image_save = None,image_dir=None, up_layer = False):
	# print("Processing {} layer...".format(name))
	how_many_features_map = activation.shape[3]
	# print('*************************************')
	# print(how_many_features_map)
	# print('*************************************')

	figure_size = how_many_features_map * 2
	fig = plt.figure(figsize=(figure_size, figure_size),)
	plt_col = 8

	if how_many_features_map == 1:
		nrows,ncol = 1,1
	elif how_many_features_map<plt_col:
		nrows,ncol = 1, how_many_features_map
	elif how_many_features_map% plt_col == 0:
		nrows,ncol = how_many_features_map // plt_col, plt_col
	else:
		nrows,ncol = how_many_features_map // plt_col + 1, plt_col

	if nrows == 1 and ncol ==1:
		fig = plt.figure()

		image = np.squeeze(activation[0, :, :, :])
		image = np.clip(image,0,1)
		plt.imshow(image)
		if up_layer == True:
			print(np.max(activation[0, :, :, :]))

			assert np.max(image) <= 1
			assert np.min(image) >= 0
		plt.clim(0,1)
		plt.colorbar()
		


	else:
		grid = ImageGrid(fig, 111,
						 nrows_ncols=(nrows, ncol),
						 axes_pad=0.1,  # pad between axes in inch.
						 )
		# print('activation shape', activation.shape)
		images = [activation[0, :, :, i] for i in range(how_many_features_map)]

		for ax, img in zip(grid, images):
			# Iterating over the grid returns the Axes.
			ax.matshow(img)

	if image_save != None:
		fig.savefig(os.path.join(image_dir,  '{}.jpeg'.format(name)))
	if up_layer == True:
		return np.squeeze(activation[0, :, :, :])
	else:
		return None
	# plt.show()


def visualize_transformation_hardcode(model,inp,layer_track,sub_model= None,image_save=None,n_image=5,all_ells=None,transform_type = 'actor'):
	#https://taiolifrancesco.medium.com/what-a-cnn-see-visualizing-intermediate-output-of-the-conv-layers-with-tensorflow-f935f55f9e8d

	if not os.path.exists(image_save):
		os.mkdir('visualize_transformation')


	all_image = []
	all_orig = []
	for j in range(n_image):
		print("Processing image {}...".format(str(j)))
		stim_image = inp[[j],:,:]
		image_dir = os.path.join(image_save,'visualize_transformation_{}'.format(transform_type),'image_%d'%(j))

		if not os.path.isdir(image_dir):
			os.makedirs(image_dir)



		fig = plt.figure()
		plt.imshow(np.squeeze(stim_image[:,:]))
		plt.colorbar()
		all_orig.append(stim_image[:,:])

		if image_save != None:
			fig.savefig(os.path.join(image_dir,  'stim.jpeg'))
		# stim_image = np.expand_dims(stim_image, axis=-1)

		# print(model.layers)
		# print(stim_image.shape)
		actor_output = model.return_actor_image(stim_image)

		plt.close('all')
		all_image.append(actor_output)
	return all_image,all_orig


def visualize_kernel(model):

	weights = model.layers[1].get_weights()[0]
	filters = weights
	# Number of filters in the conv layer
	num_filters = filters.shape[3]

	# Set up the grid for subplots
	num_columns = 3
	num_rows = 2

	fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2))

		# Plot each filter
	cbar_lim = [(-0.08,0.02),(-0.06,0.01),(-0.02,0.08),(-0.03,0.01),(-0.01,0.04),(-3e-5,5e-5)]	
	n_bin = [5,7,5,4,5,8]


	for i in range(num_filters):
		# Get the current subplot row and column
		row = i // num_columns
		column = i % num_columns

		ax = axes[row, column]
		# Get the filter
		f = filters[:, :, :, i]
		
		# Check if the filters are single or multi-channel
		# Assuming the filters are square
		if f.shape[2] == 1:
			# If single channel (grayscale), plot them in gray scale
			im = ax.imshow(f[:, :, 0], cmap='gray',vmin = cbar_lim[i][0],vmax = cbar_lim[i][1])

		else:
			raise ValueError("Kernel has multiple channels. Can't plot them.")

		# Remove axis ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(f'Kernel {i+1}')
		
		cb = plt.colorbar(im, ax=ax)
		cb.locator = MaxNLocator(nbins=n_bin[i])
		cb.update_ticks()
		# curr_ticks = cb.get_ticks()
		# cb.set_ticks(np.append(curr_ticks,[cbar_lim[i][0], cbar_lim[i][1]]))

	plt.suptitle('Actor network kernels')
	plt.tight_layout()
	return fig,ax







# def mexican_hat(t, a, sigma):
#     return a * (1 - (t ** 2) / sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2))

# def difference_of_gaussians(t, a, b, mu, sigma1, sigma2):
#     return a * np.exp(-(t - mu) ** 2 / (2 * sigma1 ** 2)) - b * np.exp(-(t - mu) ** 2 / (2 * sigma2 ** 2))


def visualize_kernel_with_line(model,final_fig_dir):
	from scipy.optimize import curve_fit


	weights = model.layers[1].get_weights()[0]

	# Normalize the filters between 0-1 for visualization
	# min_val = weights.min()
	# max_val = weights.max()
	# filters = (weights - min_val) / (max_val - min_val)

	filters = weights
	# Number of filters in the conv layer
	num_filters = filters.shape[3]

	# Set up the grid for subplots
	# num_columns = 3
	# num_rows = 2

	# fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2))

		# Plot each filter
	cbar_lim = [(-0.08,0.02),(-0.06,0.01),(-0.02,0.08),(-0.03,0.01),(-0.01,0.04),(-3e-5,5e-5)]	
	n_bin = [5,7,5,4,5,8]
	n_steps = [0.01,0.01,0.01,0.01,0.01,1e-5]
	for i in range(num_filters):
		# Get the filter
		f = filters[:, :, :, i]
		
		fig = plt.figure(figsize=(10,10))

		gs = fig.add_gridspec(2,2, width_ratios=[4, 1], height_ratios=[1, 4],
						left = 0.1, right = 0.9, bottom = 0.1, top = 0.9,
						wspace=0.05, hspace=0.05)
		
		ax = fig.add_subplot(gs[1,0])
		ax_linex = fig.add_subplot(gs[0,0],sharex=ax)
		ax_liney = fig.add_subplot(gs[1,1],sharey=ax)
		cbar_ax = fig.add_subplot(gs[0,1])

		ax_linex.set_xticks([])
		# ax_linex.set_yticks(np.linspace(cbar_lim[i][0],cbar_lim[i][1], num = n_bin[i]))
		ax_linex.set_yticks(cbar_lim[i])
		ax_linex.set_ylim(cbar_lim[i])
		# ax_linex.set_yticks(np.arange(cbar_lim[i][0],cbar_lim[i][1]+n_steps[i], 0.01))
		
		# ax_liney.set_xticks(np.linspace(cbar_lim[i][0],cbar_lim[i][1], num = n_bin[i]))
		ax_liney.set_xticks(cbar_lim[i])
		ax_liney.set_xlim(cbar_lim[i])
		# ax_liney.set_xticks(np.arange(cbar_lim[i][0],cbar_lim[i][1]+n_steps[i], 0.01))
		ax_liney.set_yticks([])

		cbar_ax.set_xticks([])
		cbar_ax.set_yticks([])

		im = ax.imshow(f[:, :, 0], cmap='gray',vmin = cbar_lim[i][0],vmax = cbar_lim[i][1])

		im_hor = ax_linex.plot(range(f.shape[0]),f[f.shape[0]//2,:,0],color='black')
		ax_linex.axhline(0,color='black',linestyle='--')
		

		t = np.array(range(f.shape[0])) - f.shape[0]//2


		# initial_guess = [1, 0.5, 0, 1, 2]
		# popt, pcov = curve_fit(difference_of_gaussians, t , f[f.shape[0]//2,:,0], p0=initial_guess, maxfev = 10000000,method='lm') 


		im_vert = ax_liney.plot(f[:,f.shape[1]//2,0],range(f.shape[0]),color='black')
		ax_liney.axvline(0,color='black',linestyle='--')
		# initial_guess = [1, 0.5, 0, 1, 2]
		# popt, pcov = curve_fit(difference_of_gaussians, t , f[:,f.shape[1]//2,0], p0=initial_guess, maxfev = 10000000,method='lm') 




		ax.set_xticks([])
		ax.set_yticks([])
		plt.suptitle(f'Kernel {i+1}')

		cb = plt.colorbar(im, ax=cbar_ax)
		cb.locator = MaxNLocator(nbins=n_bin[i])
		cb.update_ticks()


		
		plt.tight_layout()
		fig.savefig(os.path.join(final_fig_dir,f'kernel_{i}.svg'))
		# plt.show()





	# plt.suptitle('Actor network kernels')
	# plt.tight_layout()
	return fig,ax
