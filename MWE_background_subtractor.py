import cv2
import numpy as np


def blackOutCorners(img, percentage):
	# TL corner
	tl = (0, 0)
	tr = (0, int(percentage * (np.shape(img)[0])))
	bl = (int(percentage * (np.shape(img)[1])), 0)
	triangle_cnt = np.array([tl, tr, bl])
	print(triangle_cnt)
	cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

	# BL corner
	bl = (0, np.shape(img)[0])
	tl = (0, int(((1-percentage) * (np.shape(img)[0]))))
	br = (int((percentage) * np.shape(img)[1]), (np.shape(img)[0]))
	triangle_cnt = np.array([bl, tl, br])
	print(triangle_cnt)
	cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

	# BR corner
	br = list(reversed(np.shape(img)[0:2]))
	bl = (np.shape(img)[1], int((1-percentage) * (np.shape(img)[0])))
	tr = (int((1-percentage) * np.shape(img)[1]), (np.shape(img)[0]))
	triangle_cnt = np.array([tr, bl, br])
	print(triangle_cnt)
	cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

	# TR corner
	br = (np.shape(img)[1], int(percentage * (np.shape(img)[0])))
	bl = (np.shape(img)[1], 0)
	tl = (int((1 - percentage) * np.shape(img)[1]), 0)
	triangle_cnt = np.array([tl, bl, br])
	print(triangle_cnt)
	cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

	return img


dev = cv2.VideoCapture(0)
dev.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
dev.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
#_, background = dev.read()

while(True):

	_, cap = dev.read()
	crop_cap = cap[10:-10, 50:-50, :]
	crop_cap = blackOutCorners(crop_cap, 21/100)
	#r = int(np.shape(crop_cap)[1] / 2)
	#x = int(np.shape(crop_cap)[0] / 2)
	#y = int(np.shape(crop_cap)[1] / 2)
	#circle_mask = np.zeros((x*2, y*2), dtype=np.uint8)
	#cv2.circle(circle_mask, (x, y), r, (255, 255, 255), -1, 8, 0)

	#crop_cap = cv2.bitwise_and(crop_cap, crop_cap, mask=circle_mask)

	cap_gray = cv2.cvtColor(crop_cap, cv2.COLOR_BGR2GRAY)
	edges_orig = cv2.Canny(cap_gray, 100, 200)
	edges_orig = cv2.dilate(edges_orig, (5, 5), iterations=4)

	#imgHLS = cv2.cvtColor(background, cv2.COLOR_BGR2HLS)
	#hist, bins = np.histogram(imgHLS[0], 10)
	#print(hist)
	#print(bins)

	#maxIndex = np.argmax(hist)
	#lowerBound = bins[maxIndex]
	#higherBound = bins[maxIndex + 1]
	#print(higherBound)
	#print(lowerBound)
	#capHLS = cv2.cvtColor(cap, cv2.COLOR_BGR2HLS)
	#mask = (capHLS[:, :, 0] <= higherBound) & (lowerBound <= capHLS[:, :, 0])
	#mask = np.uint8(mask)

	#closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
	#closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (10, 10))
	#mask = cv2.blur(mask, (2, 2))
	#print(mask)
	#print(masked_data)
	#masked_data = np.where(cap == background, 0, cap)


	#print(mask)
	#print(np.shape(mask))
	#masked_data = cv2.bitwise_and(cap, cap, mask=mask)

	#gray_masked = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)
	#edges = cv2.Canny(gray_masked, 100, 200)
	im2, contours, hierarchy = cv2.findContours(edges_orig,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) != 0:
		# draw in blue the contours that were founded
		cv2.drawContours(crop_cap, contours, -1, (255, 0, 0), 1)

		# find the biggest area
		c = max(contours, key=cv2.contourArea)

		x, y, w, h = cv2.boundingRect(c)
		# draw the book contour (in green)
		cv2.rectangle(crop_cap, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow('crop_edges', edges_orig)
	cv2.imshow('crop', crop_cap)
	key = cv2.waitKey(20)
	if key == ord('q'):
		cv2.destroyAllWindows()
		break
