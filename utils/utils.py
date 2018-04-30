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

#get all image in the given directory persume that this directory only contain image files
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname+'/'+img_name for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs

def cycle_through_images(key, _imgs, index):

	n = len(_imgs)

	if key == ord('p'):
		index = index + 1
		if index >= n:
			index = 0
		print 'Next Image...'
	if key == ord('o'):
		index = index - 1
		if index <= 0:
			index = n - 1
		print 'Previous Image...'

	new_img = np.copy(_imgs[index])
	return new_img, index

def cycle_through_filters(key, index, max_index=2):

	if key == ord('l'):
		index += 1
		if index >= max_index:
			index = 0
		print 'Next Filter...'
	if key == ord('k'):
		index -= 1
		if index < 0:
			index = max_index - 1
		print 'Previous Filter...'

	filter_index = index
	return filter_index
