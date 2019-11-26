# Relative imports necessary in every MWE
import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from image_capture import ImageCapture
import cv2
import numpy as np



# Real code

camera = ImageCapture()

while True:
    img_raw = camera.capture()
    img_cut = img_raw[:,int(np.shape(img_raw)[1]*1/5):int(np.shape(img_raw)[1]*4/5),:]
    cv2.imshow('window', img_raw)
    cv2.imshow('cut_window', img_cut)
    if cv2.waitKey(20) == ord('q'):
        cv2.destroyAllWindows()
        break
