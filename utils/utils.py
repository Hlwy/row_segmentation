# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO

import numpy as np
import os
import cv2
import csv

#get all image in the given directory persume that this directory only contain image files
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname+'/'+img_name for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs, img_paths

def cycle_through_images(key, _imgs, _paths, index, flags=[False]):
	n = len(_imgs)
	_flag = False
	post_recording_step = flags[0]

	if key == ord('p') or post_recording_step == True:
		index = index + 1
		if index >= n:
			index = 0
		_flag = True
		print('Next Image...')
	if key == ord('o'):
		index = index - 1
		if index < 0:
			index = n - 1
		_flag = True
		print('Previous Image...')

	new_img = np.copy(_imgs[index])
	new_img_path = _paths[index]
	return new_img, new_img_path, index, _flag

def cycle_through_filters(key, index, max_index=2):
	if key == ord('l'):
		index += 1
		if index >= max_index:
			index = 0
		print('Next Filter...')
	if key == ord('k'):
		index -= 1
		if index < 0:
			index = max_index - 1
		print('Previous Filter...')

	filter_index = index
	return filter_index


def export_list2csv(_path, _file, _headers, _datalist):

	filenames = os.path.split(_file)
	filename = filenames[-1]
	print(filename)

	if not os.path.exists(str(_path)):
		print("Target output directory [" + str(_path) + "] does not exist --> MAKING IT NOW")
		os.makedirs(_path)

	csvFile = str(_path) + "/" + str(filename) + ".csv"
	with open(csvFile, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerow(_headers)

		for row in range(len(_datalist)):
			tmpData = _datalist[row]
			writer.writerow(tmpData)


	print("	Data exporting to ...")

def import_csv2list(_filepath):
	data = []
	with open(_filepath, 'rb') as sd:
		r = csv.DictReader(sd)
		for line in r:
			data.append(line)
	return data

def add_filename_prefixs(_dir, _prefix):
	filenames = os.listdir(_dir)
	os.chdir(_dir)
	for file in filenames:
		newName = str(_prefix) + "_" + str(file)
		os.rename(file, newName)
		# print(file)
		# print(newName)
	# print(filenames)
	print("Finished")
