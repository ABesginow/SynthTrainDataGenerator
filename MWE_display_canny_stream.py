from image_capture import ImageCapture
from image_processing import ImageProcessing
import cv2
import numpy as np


camera = ImageCapture()
#camera = cv2.VideoCapture(0)
processor = ImageProcessing()

while True:
    img_raw = camera.capture()
    img_cut = img_raw[:,int(np.shape(img_raw)[1]*1/5):int(np.shape(img_raw)[1]*4/5),:]
    img_gray = processor.gray(img_cut)
    edge = processor.canny(img_gray)
    contour = processor.max_contour(edge)
    cv2.drawContours(img_cut, contour, -1, (0,255,0), 3)
    bounding_box = processor.bounding_box(contour)
    print(bounding_box)
    #if bounding_box != -1:
    #    print("success!")
    #else:
    #    print("failure!")            
    cv2.imshow('raw_image', img_raw)
    cv2.imshow('cut_image', img_cut)
    cv2.imshow('gray_image', img_gray)
    cv2.imshow('canny_image', edge)
    cv2.waitKey(20)
