import shutil
import numpy as np
import cv2
import glob, os


folder = 'new_folder/'
train_txt = 'train.txt'
valid_txt = 'test.txt'
images_path = folder + "data/"


all_image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
# size of validation set in percent
validation_size = 0.2

# validation
validation_images = all_image_files[:int(len(all_image_files)*validation_size)]

# training
training_images = all_image_files[int(len(all_image_files)*validation_size):]

with open(train_txt) as fp:
    for image in training_images:
        fp.write(image + "\n")

with open(valid_txt) as fp:
    for image in validation_images:
        fp.write(image + "\n")

