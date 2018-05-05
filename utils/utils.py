# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO

import numpy as np
import Image
import os
import cv2

#get all image in the given directory persume that this directory only contain image files
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname+'/'+img_name for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs, img_paths

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


def fig2data(fig):
	"""
	@brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
	@param fig a matplotlib figure
	@return a numpy 3D array of RGBA values
	"""
	# draw the renderer
	fig.canvas.draw()

	# Get the RGBA buffer from the figure
	w,h = fig.canvas.get_width_height()
	buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
	buf.shape = (w, h,4)

	# canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
	buf = numpy.roll(buf, 3, axis = 2)
	return buf

def fig2img(fig):
	"""
	@brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
	@param fig a matplotlib figure
	@return a Python Imaging Library ( PIL ) image
	"""
	# put the figure pixmap into a numpy array
	buf = fig2data(fig)
	w, h, d = buf.shape
	return Image.fromstring("RGBA",(w ,h), buf.tostring( ))
