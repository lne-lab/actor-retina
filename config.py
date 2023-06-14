#Get information about file path and stim path

import numpy as np



def get_file_numbers_actor_highres():
	#return the file numbers for WN, Nat, and paired
	wn_files = set(np.arange(0,64,16))
	sta_files = np.arange(64,79)
	flash_file = 80
	files_highres=set(np.arange(0,64)) - wn_files
	return list(wn_files),list(files_highres), sta_files,flash_file

def get_file_numbers_5res():
	#return the file numbers for multires experiments
	wn_files = np.arange(0,64,16)
	sta_files = np.arange(64,76)
	flash_file = 76

	files_512, files_256, files_128, files_64, files_32 = [], [], [], [], []

	for files in ([1,6,11]):
		files_512.extend(np.arange(files,64,16))
	for files in ([2,7,12]):
		files_256.extend(np.arange(files,64,16))
	for files in ([3,8,13]):
		files_128.extend(np.arange(files,64,16))
	for files in ([4,9,14]):
		files_64.extend(np.arange(files,64,16))
	for files in ([5,10,15]):
		files_32.extend(np.arange(files,64,16))

	files_512.sort()
	files_256.sort()
	files_128.sort()
	files_64.sort()
	files_32.sort()

	return wn_files, files_512, files_256, files_128, files_64, files_32 ,sta_files,flash_file
	
def get_file_numbers_validation():
	#return the file numbers for WN, Nat, and paired
	wn_files = np.arange(0,125,25)
	sta_files = np.arange(125,155)
	flash_file = 155

	orig_1, orig_2, act_32, act_64, avg_32, avg_64 = [], [], [], [], [], []
	n=125
	steps=25

	for files in ([1,7,13,19]):
		orig_1.extend(np.arange(files,n,steps))
	for files in ([2,8,14,20]):
		orig_2.extend(np.arange(files,n,steps))
	for files in ([3,9,15,21]):
		act_32.extend(np.arange(files,n,steps))
	for files in ([4,10,16,22]):
		act_64.extend(np.arange(files,n,steps))
	for files in ([5,11,17,23]):
		avg_32.extend(np.arange(files,n,steps))
	for files in ([6,12,18,24]):
		avg_64.extend(np.arange(files,n,steps))

	orig_1.sort()
	orig_2.sort()
	act_32.sort()
	act_64.sort()
	avg_32.sort()
	avg_64.sort()

	return wn_files, orig_1, orig_2, act_32, act_64, avg_32, avg_64 ,sta_files,flash_file
