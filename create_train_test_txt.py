import shutil
import numpy as np
import cv2
import glob, os
from random import shuffle

# size of validation set in percent
validation_size = 0.2

folder = 'new_folder/'
train_txt = 'train.txt'
valid_txt = 'test.txt'
images_path = folder + "data/"

cwd = os.getcwd()
result = glob.glob(images_path)
shuffle(result)

with open("train.txt", 'w') as f:
    for elem in result[:len(result)*(1-validation_size)]:
        f.write(elem + "\n")
with open("test.txt", 'w') as f:
    for elem in result[len(result)*(1-validation_size):]:
        f.write(elem + "\n")



