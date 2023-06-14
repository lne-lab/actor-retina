import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import scipy.signal
from scipy.stats import ttest_rel,wilcoxon, shapiro
import seaborn as sns

def calc_local_contrast(img,window):
	mean_op = np.ones((window,window))/(window*window)
	mean_of_sq = scipy.signal.convolve2d(img**2,mean_op,mode='same',boundary='symm')
	sq_of_mean = scipy.signal.convolve2d(img,mean_op,mode='same',boundary='symm')**2
	win_var = mean_of_sq - sq_of_mean
	return np.mean(win_var)


orig_images = np.zeros([200])
act_images_32 = np.zeros([200])
act_images_64 = np.zeros([200])
avg_images_32 = np.zeros([200])
avg_images_64 = np.zeros([200])

validation_image_folder_32 = r'F:\Dataset_public\models\actor_model\best_run\visualize_transformation_difference\*'
final_fig_dir = r'F:\Dataset_public\figures\figure_5'
os.makedirs(final_fig_dir,exist_ok=True)

folders_32 = glob.glob(validation_image_folder_32)
for i,folder in enumerate(folders_32):
    orig_images[i] = calc_local_contrast(plt.imread(os.path.join(folder,'orig_image.png')),7)
    act_images_32[i] = calc_local_contrast(plt.imread(os.path.join(folder,'act_image.png')),7)
    avg_images_32[i] = calc_local_contrast(plt.imread(os.path.join(folder,'avg_image.png')),7)


results = pd.DataFrame()
results['Act32 - avg32'] = act_images_32 - avg_images_32
cc,p6 = wilcoxon(act_images_32,avg_images_32,alternative = 'greater')


WIDTH_SIZE=5
HEIGHT_SIZE=9
fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))

sns.boxplot(results)
plt.savefig(os.path.join(final_fig_dir,'local_contrast.jpeg'))
