from image_capture import ImageCapture
import cv2
import numpy as np


camera = ImageCapture()

while True:
    img_raw = camera.capture()
    img_cut = img_raw[:,int(np.shape(img_raw)[1]*1/5):int(np.shape(img_raw)[1]*4/5),:]
    cv2.imshow('window', img_raw)
    cv2.imshow('cut_window', img_cut)
    cv2.waitKey(20)
