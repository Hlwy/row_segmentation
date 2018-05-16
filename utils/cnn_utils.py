import errno
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

def crop(image, top_percent, bottom_percent):
    """
    Crops an image according to the given parameters
    :param image: source image
    :param top_percent:
        The percentage of the original image will be cropped from the top of the image
    :param bottom_percent:
        The percentage of the original image will be cropped from the bottom of the image
    :return:
        The cropped image
    """
    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def resize(image, new_dim):
    """
    Resize a given image according the the new dimension
    :param image:
        Source image
    :param new_dim:
        A tuple which represents the resize dimension
    :return:
        Resize image
    """
    return scipy.misc.imresize(image, new_dim)


def random_flip(image, steering_angle, flipping_prob=0.5):
    """
    Based on the outcome of an coin flip, the image will be flipped.
    If flipping is applied, the steering angle will be negated.
    :param image: Source image
    :param steering_angle: Original steering angle
    :return: Both flipped image and new steering angle
    """
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def random_gamma(image):
    """
    Random gamma correction is used as an alternative method changing the brightness of
    training images.
    http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    :param image:
        Source image
    :return:
        New image generated by applying gamma correction to the source image
    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range=200):
    """
    Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    :param image:
        Source image on which the shear operation will be applied
    :param steering_angle:
        The steering angle of the image
    :param shear_range:
        Random shear between [-shear_range, shear_range + 1] will be applied
    :return:
        The image generated by applying random shear on the source image
    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def random_rotation(image, steering_angle, rotation_amount=15):
    """
    :param image:
    :param steering_angle:
    :param rotation_amount:
    :return:
    """
    angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
    rad = (np.pi / 180.0) * angle
    return rotate(image, angle, reshape=False), steering_angle + (-1) * rad


def min_max(data, a=-0.5, b=0.5):
    """
    :param data:
    :param a:
    :param b:
    :return:
    """
    data_max = np.max(data)
    data_min = np.min(data)
    return a + (b - a) * ((data - data_min) / (data_max - data_min))


def generate_new_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.1):
    """
    :param image:
    :param steering_angle:
    :param top_crop_percent:
    :param bottom_crop_percent:
    :param resize_dim:
    :param do_shear_prob:
    :param shear_range:
    :return:
    """
    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)

    image = crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = random_flip(image, steering_angle)

    image = random_gamma(image)

    image = resize(image, resize_dim)

    return image, steering_angle


def get_next_image_files(_log_path, batch_size=64):
    """
    The simulator records three images (namely: left, center, and right) at a given time
    However, when we are picking images for training we randomly (with equal probability)
    one of these three images and its steering angle.
    :param batch_size:
        Size of the image batch
    :return:
        An list of selected (image files names, respective steering angles)
    """
    data = pd.read_csv(_log_path)
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        img = data.iloc[index]['image'].strip()
        angle = data.iloc[index]['steering']
        image_files_and_angles.append((img, angle))

    return image_files_and_angles


def generate_next_batch(_img_path,batch_size=16):
    """
    This generator yields the next training batch
    :param batch_size:
        Number of training images in a single batch
    :return:
        A tuple of features and steering angles as two numpy arrays
    """
    while True:
        X_batch = []
        y_batch = []
        images = get_next_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(_img_path + img_file)
            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)


def save_model(model, model_name='model.json', weights_name='model.h5'):
    """
    Save the model into the hard disk
    :param model:
        Keras model to be saved
    :param model_name:
        The name of the model file
    :param weights_name:
        The name of the weight file
    :return:
        None
    """
    silent_delete(model_name)
    silent_delete(weights_name)

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)


def silent_delete(file):
    """
    This method delete the given file from the file system if it is available
    Source: http://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
    :param file:
        File to be deleted
    :return:
        None
    """
    try:
        os.remove(file)

    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

def gen_nvidia_cnn_model():
	activation_relu = 'relu'

	# Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
	# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	model = Sequential()

	model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

	# starts with five convolutional and maxpooling layers
	model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Flatten())

	# Next, five fully connected layers
	model.add(Dense(1164))
	model.add(Activation(activation_relu))

	model.add(Dense(100))
	model.add(Activation(activation_relu))

	model.add(Dense(50))
	model.add(Activation(activation_relu))

	model.add(Dense(10))
	model.add(Activation(activation_relu))

	model.add(Dense(4))

	model.summary()
	return model

def create_model(hd5Path):
	activation_relu = 'relu'

	model = Sequential()

	model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

	# starts with five convolutional and maxpooling layers
	model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Activation(activation_relu))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Flatten())

	# Next, five fully connected layers
	model.add(Dense(1164))
	model.add(Activation(activation_relu))

	model.add(Dense(100))
	model.add(Activation(activation_relu))

	model.add(Dense(50))
	model.add(Activation(activation_relu))

	model.add(Dense(10))
	model.add(Activation(activation_relu))

	model.add(Dense(4))

	model.compile(optimizer='adam', loss='mse')

	model.load_weights(hd5Path)

	return model

def import_ground_truth_log(_log_path):

	ground_truth = []
	data = pd.read_csv(_log_path)
	nData = len(data)
	# rnd_indices = np.random.randint(0, num_of_img, batch_size)

	# for index in rnd_indices:
	for index in range(nData):
		img = data.iloc[index]['image'].strip()
		mLeft = data.iloc[index]['mLeft']
		bLeft = data.iloc[index]['bLeft']
		mRight = data.iloc[index]['mRight']
		bRight = data.iloc[index]['bRight']
		ground_truth.append((img, mLeft, bLeft, mRight,bRight))

	# print(ground_truth)
	return ground_truth

def extract_ground_truth_log(_log_path):
	while True:
		X_batch = []
		y_batch = []
		ground_truth = import_ground_truth_log(_log_path)

		for img_file, mLeft, bLeft, mRight,bRight in ground_truth:
			# raw_image = plt.imread(img_file)
			raw_image = cv2.imread(img_file)
			targets = [mLeft, bLeft, mRight,bRight]
			new_image = resize(raw_image, (64, 64))
			X_batch.append(new_image)
			y_batch.append(targets)


		yield np.array(X_batch), np.array(y_batch)
