#!/bin/bash

#SBATCH --job-name=row_segmentation
#SBATCH --partition=terramepp
#SBATCH --gres=gpu:4
#SBATCH --mail-user=huntery2@illinois.edu

module load OpenCV/3.3.0-IGB-gcc-4.9.4-Python-3.6.1
module load Keras/2.0.8-IGB-gcc-4.9.4-Python-3.6.1
module load CUDA/8.0.61-IGB-gcc-4.9.4
module load Tensorflow-GPU/1.2.1-IGB-gcc-4.9.4-Python-3.6.1

python cnn_train_hsv.py
