import os
import cv2
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from utils import cnn_utils as helper

number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 3200
learning_rate = 1e-4

trainPath = os.getcwd() + "/exported/small_set_training_log.csv"
testPath = os.getcwd() + "/exported/validate_training_log.csv"

model = helper.gen_nvidia_cnn_model()
model.compile(optimizer=Adam(learning_rate), loss="mse", )

# create two generators for training and validation
train_gen = helper.extract_ground_truth_log(trainPath)
validation_gen = helper.extract_ground_truth_log(testPath)

history = model.fit_generator(train_gen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

# finally save our model and weights
helper.save_model(model)
