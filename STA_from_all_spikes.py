import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import os
import config
import utils_sta as utils_STA
import cv2
import glob
from pathlib import Path

expt_type = 'validation'

expt_root = Path(r'F:\Retina_project\Dataset_public\STA_images\validation_used')

st_folders = glob.glob(str(expt_root/'*'))

total_neurons_pass = 0
total_neurons = 0

st_root = r'F:\Retina_project\Dataset_public\STA_images\validation_used\STA'
os.makedirs(st_root,exist_ok=True)
neuron_all_id = 0
neuron_RF = pd.DataFrame(columns=['folder_name','neuron_id','x','y','major_sigma1','minor_sigma1','angle','area_sigma1','cicular_diameter_sigma1','major_sigma2','minor_sigma2','area_sigma2','circular_diameter_sigma2','on_off'])




def on_trackbar(val):
	global threshold_value
	threshold_value = val
	_, thresh = cv2.threshold(adjusted, threshold_value, 255, cv2.THRESH_BINARY)

	two_image = np.concatenate([adjusted,thresh],axis=1)
	cv2.imshow('Threshold', two_image)

stim_path = r'F:\Retina_project\Dataset_public\stimulus\validation_stim'
wn_files, orig_1, orig_2, act_32, act_64, avg_32, avg_64 ,sta_files,flash_file = config.get_file_numbers_validation()

all_spikes = np.load(r'F:\Retina_project\Dataset_public\spikes_data\validation_spikes_full.npy')
st_folder = all_spikes
# print(all_spikes.shape)
# print(neuron_pass)

# for n in neuron_pass:
for n in range(all_spikes.shape[1]):
	try:
		print('Neuron: ', neuron_all_id)
		sta = utils_STA.compute_STA_splitfiles(np.squeeze(all_spikes[:,n,:]),stim_path,20,sta_files,to_norm=True)
		# sta = np.array(sta)
		spatial,timecourse = utils_STA.decompose(sta)
		orig = spatial
		plt.close('all')

		fig, ax = plt.subplots()
		kwargs = {"fig": fig,"ax":ax}
		utils_STA.spatial(spatial,**kwargs, color='seismic')
		save_folder_name = 'sta_analysis_color'
		os.makedirs(os.path.join(st_root,save_folder_name),exist_ok=True)
		# if not os.path.isdir(os.path.join(st_root,save_folder_name)):
		# 	os.mkdir(os.path.join(st_root,save_folder_name))
		fig.savefig(os.path.join(st_root,save_folder_name,  'sta_{}_spatial.png'.format(neuron_all_id)))
        
		fig, ax = plt.subplots()
		kwargs = {"fig": fig,"ax":ax}
		utils_STA.spatial(spatial,color = 'gray',**kwargs)
		save_folder_name = 'sta_analysis_gray'
		os.makedirs(os.path.join(st_root,save_folder_name),exist_ok=True)
		fig.savefig(os.path.join(st_root,save_folder_name,  'sta_{}_spatial.png'.format(neuron_all_id)))


		# Find the min and max pixel values
		min_pixel = np.min(spatial)
		max_pixel = np.max(spatial)

		# Perform min-max scaling to [0, 255]
		spatial = (spatial - min_pixel) * (255.0 / (max_pixel - min_pixel))
		spatial = spatial.astype(np.uint8)

		adjusted = cv2.convertScaleAbs(spatial, alpha=1.5)
		adjusted = cv2.medianBlur(adjusted, 3)
		adjusted = cv2.resize(adjusted,(640,640))
		spatial_big = cv2.resize(orig,(640,640),interpolation=cv2.INTER_NEAREST)
		# print(adjusted)

		cv2.namedWindow('Threshold',cv2.WINDOW_FULLSCREEN)
		cv2.createTrackbar('thresh', 'Threshold', 0, 255, on_trackbar)
		on_trackbar(0)
		key = cv2.waitKey(0)
		cv2.destroyAllWindows()
		if key == ord('w'):
			thresh_method = cv2.THRESH_BINARY
			on_off_type = 1
		elif key == ord('s'):
			thresh_method = cv2.THRESH_BINARY_INV
			on_off_type = 0
		elif key == ord('n'):
			neuron_RF.loc[neuron_all_id] = [st_folder,n,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
			neuron_RF.to_csv(os.path.join(st_root,'neuron_RF.csv'))
			neuron_all_id += 1

			continue
		_, thresh = cv2.threshold(adjusted, threshold_value, 255, thresh_method)
		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		largest_contour = max(contours, key=cv2.contourArea)
		mask = np.zeros_like(adjusted)
		cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
		thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        
		adjusted=thresh


		# fig, ax = plt.subplots()
		# # plt.imshow(stan)
        
		# kwargs = {"fig": fig,"ax":ax}
		# utils_STA.spatial(spatial_big,**kwargs, color='seismic')
		# # utils_STA.ellipse(adjusted,**kwargs)
		# # plt.show()
		# save_folder_name = 'sta_analysis_adjusted'
		# if not os.path.isdir(os.path.join(st_root,save_folder_name)):
		# 	os.mkdir(os.path.join(st_root,save_folder_name))
		# fig.savefig(os.path.join(st_root,save_folder_name,  'sta_{}_{}_spatial.png'.format(n,good_neurons[n])))


		# fig = plt.figure()
		# plt.imshow(adjusted)
        
		# stan = (spatial - np.mean(spatial))/np.var(spatial)
		# fig, ax = plt.subplots()
		# plt.imshow(adjusted)
		# kwargs = {"fig": fig,"ax":ax}
		# utils_STA.spatial(adjusted,color = 'gray',**kwargs)
		fig, ax = plt.subplots()
		kwargs = {"fig": fig,"ax":ax}
		flipped_image = cv2.flip(spatial_big, 0)

		utils_STA.spatial(flipped_image,**kwargs, color='seismic')
		# utils_STA.ellipse(adjusted,sigma=0.5,**kwargs)
        

		# _, thresh = cv2.threshold(adjusted, threshold_value, 255, cv2.THRESH_BINARY)
		# print(thresh.shape)
		utils_STA.ellipse(adjusted,sigma=2.,**kwargs)
		center,widths_sigma2, theta = utils_STA.get_ellipse(adjusted,sigma=2)
		center, widths_sigma2 = map(lambda x: np.asarray(x) * 5, (center, widths_sigma2))
		RF_area_sigma2 = np.pi*widths_sigma2[0]*widths_sigma2[1]/(4*1e6)

		center,widths_sigma1, theta = utils_STA.get_ellipse(adjusted,sigma=1)
		center, widths_sigma1 = map(lambda x: np.asarray(x) * 5, (center, widths_sigma1))
		# print(center,widths_sigma1, theta)
		RF_area_sigma1 = np.pi*widths_sigma1[0]*widths_sigma1[1]/(4*1e6)
		neuron_RF.loc[neuron_all_id] = [st_folder,n,center[0],center[1],widths_sigma1[0],widths_sigma1[1],theta,RF_area_sigma1,2*np.sqrt(RF_area_sigma1/np.pi),widths_sigma2[0],widths_sigma2[1],RF_area_sigma2,2*np.sqrt(RF_area_sigma2/np.pi),on_off_type]
		print(neuron_RF.loc[neuron_all_id])
		# contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		# largest_contour = max(contours, key=cv2.contourArea)
		# ellipse = cv2.fitEllipse(largest_contour)
		# cv2.ellipse(adjusted, ellipse, (0,255,0), 2)
		# plt.imshow(adjusted)
		# plt.show()
		save_folder_name = 'sta_analysis_ellipse'
		os.makedirs(os.path.join(st_root,save_folder_name),exist_ok=True)
		fig.savefig(os.path.join(st_root,save_folder_name,  'sta_{}_spatial.png'.format(neuron_all_id)))
		neuron_all_id += 1

		#save neuron_rf
		neuron_RF.to_csv(os.path.join(st_root,'neuron_RF.csv'))

		# spatial -= np.mean(spatial)
		# os.makedirs(os.path.join(st_root,save_folder_name),exist_ok=True)
		# np.save(os.path.join(st_root,save_folder_name,'sta_{}_{}.npy'.format(n,good_neurons[n])),spatial)

	except Exception as e:
		print(e)
        
		neuron_RF.loc[neuron_all_id] = [st_folder,n,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
		neuron_RF.to_csv(os.path.join(st_root,'neuron_RF.csv'))
		neuron_all_id += 1
		continue