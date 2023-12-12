import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import os
import utils,config
import glob
from scipy.stats import bootstrap
from scipy import stats
from scipy.stats import ttest_rel,wilcoxon, shapiro
def parallel_axis_dot(befores,afters,labels,title=None,save_image=None):

	save_type = 'svg'
	# plotting the points
	fig = plt.figure(figsize=(5,9))
	plt.scatter(np.zeros(len(befores)), befores, c='grey',alpha = 0.2,s = 5)
	plt.scatter(np.ones(len(afters)), afters, c='grey',alpha = 0.2,s = 5)

	plt.scatter([0], np.mean(befores), c='k',s = 50)
	plt.scatter([1], np.mean(afters), c='k',s=50)
	plt.plot( [0,1], [np.mean(befores),np.mean(afters)], c='k')

	# plotting the lines
	for i in range(len(befores)):
	    plt.plot( [0,1], [befores[i], afters[i]], c='grey',alpha = 0.2)

	plt.xticks([0,1], labels)
	plt.title(title)
	fig.savefig(os.path.join(save_image,'%s_scatter.%s'%(title,save_type)))

	fig = plt.figure()
	plt.boxplot(np.array(afters)-np.array(befores))
	plt.axhline(0,c='k',ls='--')
	plt.title(title)
	fig.savefig(os.path.join(save_image,'%s_box.%s'%(title,save_type)))
	return np.array(afters)-np.array(befores)


def odd_even_avg_response(spikes,files,num_unique_image = 500, num_reps = 10):
	#averages the response across the repeats to the same stimulus
	all_neuron_odd_response = np.empty((len(files)*(num_unique_image//num_reps), spikes.shape[1]))
	all_neuron_even_response = np.empty((len(files)*(num_unique_image//num_reps), spikes.shape[1]))

	odd_ind = [1,3,5,7,9]
	even_ind = [0,2,4,6,8]

	for neuron in range(spikes.shape[1]):
		neuron_odd_response = []
		neuron_even_response = []
		for file_ind in range(len(files)):
			tmp = (file_ind)*num_unique_image
			responses = spikes[tmp:tmp+num_unique_image, neuron]
			responses = responses.reshape((num_reps,num_unique_image//num_reps))
			odd_response = np.mean(responses[odd_ind,:],axis=0)
			even_response = np.mean(responses[even_ind,:],axis=0)

			neuron_odd_response.extend(odd_response)
			neuron_even_response.extend(even_response)

		all_neuron_odd_response[:,neuron] = neuron_odd_response
		all_neuron_even_response[:,neuron] = neuron_even_response

	return all_neuron_odd_response,all_neuron_even_response

def paired_test_analysis(higher_res,lower_res):
	# higher_res and lower_res are str of {512,256,128,64,32}
	higher_odd = res_dict['odd_%s'%(higher_res)]
	higher_even = res_dict['even_%s'%(higher_res)]
	lower_even = res_dict['even_%s'%(lower_res)]

	_,_,same_stim_relilability = utils.neuronal_CC_paired_stim(higher_odd,higher_even,neuron_pass=None,to_plot = True,save_dir = os.path.join(save_image,'same_res_%s_%s'%(higher_res,higher_res)),xlab='odd_%s'%(higher_res),ylab='even_%s'%(higher_res))
	_,_,downsampled_relilability = utils.neuronal_CC_paired_stim(higher_odd,lower_even,neuron_pass=None,to_plot = True,save_dir = os.path.join(save_image,'diff_res_%s_%s'%(higher_res,lower_res)),xlab='odd_%s'%(higher_res),ylab='odd_%s'%(lower_res))


	cc,p = wilcoxon(same_stim_relilability,downsampled_relilability,alternative='greater')
	
	reliability_dict = {'same_res_%s_%s'%(higher_res,higher_res):same_stim_relilability,'diff_res_%s_%s'%(higher_res,lower_res):downsampled_relilability}
	reliability_dict = pd.DataFrame(reliability_dict)

	diff = np.array(downsampled_relilability)-np.array(same_stim_relilability)
	utils.validation_generate_dist_plot(reliability_dict,'same_res_%s_%s'%(higher_res,higher_res),'diff_res_%s_%s'%(higher_res,lower_res),save_image,ylim=[0,1])
	reliability_dict.to_csv(os.path.join(save_image,'%s_%s.csv'%(higher_res,lower_res)))

	return 'one_tail_%s,%s'%(higher_res,lower_res), diff


wn_files, files_512, files_256, files_128, files_64, files_32 ,sta_files,flash_file = config.get_file_numbers_5res()
spikes_folder = r'F:\Retina_project\Dataset_public\spikes_data\multires_spikes.npy'
all_spikes = np.load(spikes_folder)

save_image = r'F:\Retina_project\Dataset_public\figures\figure_2'
os.makedirs(save_image,exist_ok=True)

wn_files_resp, files_512_resp, files_256_resp, files_128_resp, files_64_resp, files_32_resp = utils.sum_response_5res(all_spikes, 40, (wn_files, files_512, files_256, files_128, files_64, files_32),num_unique_image=200)

neuron_pass = utils.stability_check(wn_files_resp,wn_files,stability_thresh=0.3,num_unique_image=200)


# odd_512,even_512 = odd_even_avg_response(files_512_resp[:,neuron_pass],files_512,num_unique_image = 200, num_reps = 10)
odd_256,even_256 = odd_even_avg_response(files_256_resp[:,neuron_pass],files_256,num_unique_image = 200, num_reps = 10)
odd_128,even_128 = odd_even_avg_response(files_128_resp[:,neuron_pass],files_128,num_unique_image = 200, num_reps = 10)
odd_64,even_64 = odd_even_avg_response(files_64_resp[:,neuron_pass],files_64,num_unique_image = 200, num_reps = 10)
odd_32,even_32 = odd_even_avg_response(files_32_resp[:,neuron_pass],files_32,num_unique_image = 200, num_reps = 10)


res_dict = {'odd_256':odd_256,
			'even_256':even_256,
			'odd_128':odd_128,
			'even_128':even_128,
			'odd_64':odd_64,
			'even_64':even_64,
			'odd_32':odd_32,
			'even_32':even_32,}

res_set = ['256','128','64','32']
all_name = []
all_diff = []

for i in range(len(res_set)): 
	for j in range(1,len(res_set) - i):
		name, diff = paired_test_analysis(res_set[i],res_set[i+j])
		all_name.append(name)
		all_diff.append(diff)

data = pd.DataFrame(all_diff)
data=data.T
data.columns = all_name

plt.close('all')
fig,ax = plt.subplots()
data.plot(kind='box', title='boxplot',ax=ax)
plt.axhline(0,c='k',ls='--')
plt.title('%dms bins window'%(400))
plt.xticks(rotation = 45,ha="right") # Rotates X-Axis Ticks by 45-degrees

plt.tight_layout()
fig.savefig(os.path.join(save_image,'all_difference_in_NR_%d.svg'%(40)))
data.to_csv(os.path.join(save_image,'box_data.csv'))







