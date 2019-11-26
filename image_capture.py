import cv2
import numpy as np


class ImageCapture():

	def __init__(self, camera=0):
            self.cap = cv2.VideoCapture(camera)

	def capture(self):
            _, img = self.cap.read()
            return img

	def change_settings(self, setting, value):
	    self.cap.set(setting, value)
