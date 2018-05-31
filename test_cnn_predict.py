import os
import cv2
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from utils import filter_utils as fut
from utils import cnn_utils as helper

def test_cnn_lines():
	model = helper.create_model_lines("model.h5")
	model._make_predict_function()
	graph = tf.get_default_graph()

	img = cv2.imread('training/scenarios/plot_gap/plot_gap_frame310.jpg')
	clone = cv2.resize(img, (640,480))
	h,w,c = clone.shape
	image_array = np.asarray(img)

	# image_array = helper.crop(image_array, 0.35, 0.1)
	image_array = helper.resize(image_array, new_dim=(64, 64))

	transformed_image_array = image_array[None, :, :, :]
	outputs = np.float32(model.predict(transformed_image_array))
	print(outputs)
	print(outputs[0][1])

	ploty = np.linspace(0, w-1, w)
	plot_leftx = outputs[0][0]*ploty + outputs[0][1]
	plot_rightx = outputs[0][2]*ploty + outputs[0][3]
	cv2.line(clone,(int(plot_leftx[0]),int(ploty[0])),(int(plot_leftx[-1]),int(ploty[-1])),(0,0,255))
	cv2.line(clone,(int(plot_rightx[0]),int(ploty[0])),(int(plot_rightx[-1]),int(ploty[-1])),(255,0,0))
	return clone


def test_cnn_hsv():
	model = helper.create_model_hsv("models/hsv/test_1/model.h5")
	model._make_predict_function()
	graph = tf.get_default_graph()

	img = cv2.imread('training/scenarios/plot_gap/plot_gap_frame310.jpg')
	clone = cv2.resize(img, (640,480))
	h,w,c = clone.shape
	image_array = np.asarray(img)

	# image_array = helper.crop(image_array, 0.35, 0.1)
	image_array = helper.resize(image_array, new_dim=(64, 64))

	transformed_image_array = image_array[None, :, :, :]
	outputs = np.float32(model.predict(transformed_image_array))
	print(outputs)
	print(outputs[0][1])

	res, comp_mask = fut.filter_custom(clone,outputs)

	return clone

if __name__ == "__main__":
	clone = test_cnn_hsv()
	cv2.imshow("lines",clone)
	while True:
		key = cv2.waitKey(5) & 0xFF
		if key == ord('q'):
			break
	cv2.destroyAllWindows()
