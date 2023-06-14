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





def sum_response(all_spikes, n_bins, file_numbers, num_unique_image = 200):
	#sum up the response within the window
	#all_spikes: binned spike data with shape n_files, n_neurons x bins. bins is 10ms 
	#nbins: how many bins to sum up the data.
	#file_numbers: list corresponding file numbers for WN, Nat, Paired

	stim_change = np.arange(0,all_spikes.shape[2], all_spikes.shape[2]//num_unique_image) #return the index where a stimulus change occurs

	summed_response=np.empty((all_spikes.shape[0],all_spikes.shape[1],len(stim_change))) 
	for ind, i in enumerate(stim_change):
		summed_response[:,:,ind] = np.squeeze(np.sum(all_spikes[:,:,i:i+n_bins], axis = 2))

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
		summed_response[:,:,ind] = np.squeeze(np.sum(all_spikes[:,:,i:i+n_bins], axis = 2))

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
			plt.savefig(os.path.join(save_dir,'validation_neuron_scatter','neuron_%d'%(n),'%s_%s_scatter_R2_%0.3f.jpeg'%(xlab,ylab,R2)))

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
	g.fig.savefig(os.path.join(final_fig_dir,'validation_box_dist_plot','%s_%s.jpeg'%(first_condition,second_condition)))

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
		perc_increase = (results['actor_neuronal_R2'] - results['avg_neuronal_R2'])


	if actor_model is not None:
		cc2,p2 = wilcoxon(results['fwd_neuronal_R2'],results['actor_neuronal_R2'],alternative='less')
		cc1,p1 = wilcoxon(results['fwd_neuronal_R2'],results['avg_neuronal_R2'],alternative='less')
		cc3,p3 = wilcoxon(results['actor_neuronal_R2'],results['avg_neuronal_R2'],alternative='less')


	ext = 'jpeg'
	fig = plt.figure()
	sns.boxplot(diff_results,
		flierprops={"marker": "$\circ$"})
	plt.ylim([-0.1,0.15])
	plt.axhline(y=0,ls='--')
	if save_dir is not None:
		if not os.path.isdir(os.path.join(save_dir,'insilico_box_dist_plot')):
			os.mkdir(os.path.join(save_dir,'insilico_box_dist_plot'))
		plt.savefig(os.path.join(save_dir,'insilico_box_dist_plot','boxplot_%s_%s_%s_%s.%s'%(nout,p2,p1,p3,ext)))
	
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
	

