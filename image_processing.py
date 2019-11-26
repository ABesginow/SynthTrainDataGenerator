import cv2
import numpy as np
import argparse
import glob
import imutils

class ImageProcessing:

	def __init__(self, img):
		self.background = img.copy()
		pass

	def convert(self, size, box):
		dw = 1./size[0]
		dh = 1./size[1]
		x = (box[0] + box[1])/2.0
		y = (box[2] + box[3])/2.0
		w = box[1] - box[0]
		h = box[3] - box[2]
		x = x*dw
		w = w*dw
		y = y*dh
		h = h*dh
		return (x,y,w,h)

	def auto_canny(self, image, sigma=0.53):
		# compute the median of the single channel pixel intensities
		v = np.median(image)
		
		# apply automatic Canny edge detection using the computed median
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(image, lower, upper)

		edged = cv2.dilate(edged, (3, 3), iterations=1)

		# return the edged image
		return edged

	def black_out_corners(self, img, percentage):
		# TL corner
		tl = (0, 0)
		tr = (0, int(percentage * (np.shape(img)[0])))
		bl = (int(percentage * (np.shape(img)[1])), 0)
		triangle_cnt = np.array([tl, tr, bl])
		print(triangle_cnt)
		cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

		# BL corner
		bl = (0, np.shape(img)[0])
		tl = (0, int(((1 - percentage) * (np.shape(img)[0]))))
		br = (int((percentage) * np.shape(img)[1]), (np.shape(img)[0]))
		triangle_cnt = np.array([bl, tl, br])
		print(triangle_cnt)
		cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

		# BR corner
		br = list(reversed(np.shape(img)[0:2]))
		bl = (np.shape(img)[1], int((1 - percentage) * (np.shape(img)[0])))
		tr = (int((1 - percentage) * np.shape(img)[1]), (np.shape(img)[0]))
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

	def remove_background_youtube(self, img_raw, lower_green, upper_green):
		hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)

		#lower_green = np.array([56, 87, 0])
		#upper_green = np.array([69, 255, 255])
		    
		mask = cv2.inRange(hsv, lower_green, upper_green)
		mask_inv = cv2.bitwise_not(mask)
		
		fg = cv2.bitwise_and(img_raw, img_raw, mask=mask_inv)
		return fg, mask_inv


	def remove_background_histogram(self, img_raw):
		imgHLS = cv2.cvtColor(self.background, cv2.COLOR_BGR2HLS)
		hist, bins = np.histogram(imgHLS[0], bins=5)
		print(hist)
		print(bins)
		maxIndex = np.argmax(hist)
		lowerBound = bins[maxIndex]
		higherBound = bins[maxIndex + 1]
		#print(higherBound)
		#print(lowerBound)
		capHLS = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HLS)
		mask = (capHLS[:, :, 0] <= higherBound) & (lowerBound <= capHLS[:, :, 0])
		mask = np.uint8(mask)
		#closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
		#closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (10, 10))
		mask = cv2.blur(mask, (2, 2))
		#print(mask)
		#print(masked_data)
		#masked_data = np.where(cap == self.background, 0, cap)
		masked_data = cv2.bitwise_and(img_raw, img_raw, mask=mask)
		masked_data = cv2.dilate(masked_data, (3, 3), iterations=3)
		return masked_data

	def gray(self, img_raw):
		return cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

	def canny(self, img):
		upper = 200
		lower = 0
		img = cv2.blur(img, (2, 2))
		if np.shape(img)[-1] != 3:
			edges = cv2.Canny(img, lower, upper)
		else:
			edges = cv2.Canny(self.gray(img), lower, upper)
		#edges = cv2.dilate(edges, (10, 10), iterations=8)
		edges = cv2.dilate(edges, (3, 3), iterations=3)
		return edges

	def central_contour(self, img):
		im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		centerY = np.shape(img)[0]/2
		centerX = np.shape(img)[1]/2
		central_contour = 0
		bestX, bestY = 0, 0
		for c in contours:
			if cv2.contourArea(c) < 300:
				continue
			# compute the center of the contour
			M = cv2.moments(c)
			try:
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
			except:
				cX = 1000
				cY = 1000

			if np.sqrt((centerY-cY)**2 + (centerX-cX)**2) < np.sqrt((centerY-bestY)**2 + (centerX-bestX)**2):
				central_contour = c
				bestX = cX
				bestY = cY
		return central_contour


	def max_contour(self, img):
		im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) != 0:
			max_contour = max(contours, key = cv2.contourArea)
			return max_contour
		else:
			return -1

	def bounding_box(self, object_contour):
		x, y, w, h = cv2.boundingRect(object_contour)
		bounding_box = ((x, y), (x + w, y + h))
		return bounding_box

	def combine_edges(self, mask_edge, object_edge):
		dst = cv2.addWeighted(mask_edge, 1.0, object_edge, 0.0, 0)
		return dst

	def overlap(self, bb1, bb2, size=0): 
		"""
		Calculate the Intersection over Union (IoU) of two bounding boxes.

		Parameters
		----------
		bb1 : dict
			Keys: {'x1', 'x2', 'y1', 'y2'}
			The (x1, y1) position is at the top left corner,
			the (x2, y2) position is at the bottom right corner
		bb2 : dict
			Keys: {'x1', 'x2', 'y1', 'y2'}
			The (x, y) position is at the top left corner,
			the (x2, y2) position is at the bottom right corner

		Returns
		-------
		float
		in [0, 1]
		"""

		if size is not 0:
			x1 = bb1[0] * size[1]
			x2 = bb1[0] * size[1] + bb1[2] * size[1]
			y1 = bb1[1] * size[0]
			y2 = bb1[1] * size[0] + bb1[3] * size[0]
			bb1 = ((x1, y1), (x2, y2))
			x1 = bb2[0] * size[1]
			x2 = bb2[0] * size[1] + bb2[2] * size[1]
			y1 = bb2[1] * size[0]
			y2 = bb2[1] * size[0] + bb2[3] * size[0]
			bb2 = ((x1, y1), (x2, y2))

			
		# determine the coordinates of the intersection rectangle
		x_left = max(bb1[0][0], bb2[0][0])
		y_top = max(bb1[0][1], bb2[0][1])
		x_right = min(bb1[1][0], bb2[1][0])
		y_bottom = min(bb1[1][1], bb2[1][1])

		#print(str(x_left) + " " + str(y_top) + " " + str(x_right) + " " + str(y_bottom))
		
		if x_right < x_left or y_bottom < y_top:
			return 0.0

		# The intersection of two axis-aligned bounding boxes is always an
		# axis-aligned bounding box
		
		intersection_area = (x_right - x_left) * (y_bottom - y_top)

		iou = intersection_area

		return iou
		



	def multiple_OTL_on_background(self, otl_snippets, cls_ids, occlusion=True):
		# Idea for a different implementation:
		# I have a folder with the OTL snippets and can load them in case I want to combine them for such a purpose
		#
		# run OTL on background multiple times with every OTL snippet once
		# if occlusion is true, I don't care where you position the snippets
		# if it's false, try 5 times that you don't occlude it, else try again completely
		# return a different datatype for the bounding box which also holds the corresponding ids of the object classes
		#for i in range(len(otl_snippets)):
		#	self.OTL_on_background(otl_snippets[i])
		bb_array = []
		if occlusion:
			final, bb = self.OTL_on_background(otl_snippets[0])
			
			bb_array.append((cls_ids[0], bb))
			for i, snippet in enumerate(otl_snippets[1:]):
				temp, bb = self.OTL_on_background(snippet, background=final)
				if temp is 0 and bb is 0:
					continue
				else:
					final = temp.copy()
				# I have to check for overlap between the things and remove the 
				# overlap from the previous snippets' bb
				bb_array.append((cls_ids[i+1], bb))
		else:
			final, bb = self.OTL_on_background(otl_snippets[0])
			while final is 0 and bb is 0:
				final, bb = self.OTL_on_background(otl_snippets[0])
			bb_array.append((cls_ids[0], bb))
			for i, snippet in enumerate(otl_snippets[1:]):
				collision = False
				temp, bb = self.OTL_on_background(snippet, background=final)
				if temp is 0 and bb is 0:
					collision = True
				if collision is False:
					for b in bb_array:
						# if there is overlap, do it again
						if self.overlap(b[1], bb, np.shape(temp)) > 0:
							print("INITIAL OVERLAP")
							print(self.overlap(b[1], bb, np.shape(temp)))
							collision = True
							break
						else:
							collision = False
				while collision:
					temp, bb = self.OTL_on_background(snippet, background=final)
					if temp is 0 and bb is 0:
						collision = True
						continue
					for b in bb_array:
					# if there is overlap, do it again
						if self.overlap(b[1], bb, np.shape(temp)) > 0:
							print("LATER OVERLAP")
							print(self.overlap(b[1], bb, np.shape(temp)))
							collision = True
							continue
						else:
							collision = False
				final = temp.copy()
				bb_array.append((cls_ids[i+1], bb))
		return final, bb_array

	
	def OTL_on_background(self, otl_snippet, random_position=True, random_size=True, background=0):
		import os, random
		if background is 0:
			#background = cv2.imread('market.jpg')
			background = cv2.imread('Backgrounds/' + random.choice(os.listdir("Backgrounds")))
		background = imutils.resize(background, width=1000)
		
		img1 = background
		if random_size:
			OTL_size_factor = random.random()*2
			try:
				otl_snippet = imutils.resize(otl_snippet, height=int(otl_snippet.shape[0]*OTL_size_factor))
			except Exception as e:
				print(e)
				print(OTL_size_factor)
				return 0, 0

		img2 = otl_snippet

		# I want to put logo on top-left corner, So I create a ROI
		rows,cols,channels = img2.shape

		height, width, _ = img1.shape

		offset_x = 0
		offset_y = 0

		OTL_size_factor = 1
		i = 0
		if random_position:
			offset_x = int(width*random.random())
			offset_y = int(height*random.random())
			while offset_x + cols > width and i < 50:
				i += 1
				offset_x = int(width*random.random())
			while offset_y + rows > height and i < 50:
				i += 1
				offset_y = int(height*random.random())
			if i == 50:
				return 0, 0
		
		roi = img1[offset_y:offset_y + rows, offset_x:offset_x + cols]
		# Now create a mask of logo and create its inverse mask also
		img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)

		# Now black-out the area of logo in ROI
		img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

		# Take only region of logo from logo image.
		img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

		# Put logo in ROI and modify the main image
		dst = cv2.add(img1_bg,img2_fg)
		
		img1[offset_y:offset_y + rows, offset_x:offset_x + cols] = dst

		b = (float(offset_x), float(offset_x + cols), float(offset_y), float(offset_y + rows))
		bb = self.convert((width, height), b)
		
		return img1, bb

	#def transformation(self, transformation, bounding_box, img_raw, background_folder):
	#       return
