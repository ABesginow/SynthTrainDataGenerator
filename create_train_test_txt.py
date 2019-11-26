import shutil
import numpy as np
import cv2
import glob, os


folder = 'new_folder/'
train_txt = 'train.txt'
valid_txt = 'test.txt'
images_path = folder + "data/"


all_image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]

# 10% for validation
validation_images = all_image_files[:int(len(all_image_files)*0.1)]
# 90% for training
training_images = all_image_files[int(len(all_image_files)*0.1):]
with open(train_txt) as fp:
    for image in training_images:
        fp.write(image + "\n")

with open(valid_txt) as fp:
    for image in validation_images:
        fp.write(image + "\n")

