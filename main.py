"""

Main file
Purpose:
-> Executes the main loop where images are taken and given to threads to execute edge detection and contour detection




structure of main loop:

>initialise variables
|>capture image
|>start thread
||>increase contrast
||>run edge detection
||>determine largest contour of the object
||>determine outer box of largest contour
||>safe image to correct folder
||>write information in txt file in correct yolo format
|>move motor --> loop back to image capture
>loop is finished as soon as 12360 iterations have been executed (1 full circle)
>write everything into files (image names, txt names etc.)
>zip everything
>send everything over network to alienware
>remote start script on alienware
>wait
"""

from image_capture import ImageCapture
from motor_control_PI import MotorControl
from image_processing import ImageProcessing
from file_operations import FileOperations
#from CLI import CLI
import cv2
import numpy as np
import hashlib
import time
import os
import random
import shutil

RPI_CAMERA = 0
IMAGE_FOLDER = "Images/"
BOX_FOLDER = "Boxes/"


def nothing(x):
	pass

def save_to_files(bounding_box, final):
	filename = str(hashlib.md5(str.encode(str(time.time()))).hexdigest())
	# folder for the class
	foldername = "new_folder/"
	# datafolder for the images and the txt files
	datafolder = foldername + "data/"
	cwd = os.getcwd()
	newpath = cwd + "/" + datafolder
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	imagefile = datafolder + filename + '.png' 
	textfile = open(datafolder + filename + '.txt', 'a')
	
	file_operations.save_to_folder(textfile, imagefile, bounding_box, final)

	

# todo get classes through GUI
classes = []

#interface = CLI()
file_operations = FileOperations()
motor = MotorControl()
camera = ImageCapture(RPI_CAMERA)
image_processor = ImageProcessing(camera.capture())


delay = 1/1000.0
#images = input("How many images do you want per category (5 categories)?")
images = 3000
STEPS_FOR_FULL_CIRCLE = 12360
steps = int(STEPS_FOR_FULL_CIRCLE/images)
classes = ["Halterung1", "Halterung2", "Rolle"]

## Section for the configuration
for label in classes:
	last_five_OTL_sizes = [0]*5
	panel = np.zeros([100, 700], np.uint8)
	cv2.namedWindow('panel')

	# 056
	cv2.createTrackbar('L - h', 'panel', 0, 179, nothing)
	# 069
	cv2.createTrackbar('U - h', 'panel', 179, 179, nothing)

	# 087
	cv2.createTrackbar('L - s', 'panel', 0, 255, nothing)
	# 255
	cv2.createTrackbar('U - s', 'panel', 255, 255, nothing)

	# 0
	cv2.createTrackbar('L - v', 'panel', 0, 255, nothing)
	# 255
	cv2.createTrackbar('U - v', 'panel', 255, 255, nothing)
	
	switch = '0 : OFF \n1 : ON'
	cv2.createTrackbar(switch, 'panel',0,1,nothing)
	string_motor = '0 : MOTOR OFF \n1 : MOTOR ON'
	cv2.createTrackbar(string_motor, 'panel',0,1,nothing)

	while True:
		img_raw = camera.capture()
	
		l_h = cv2.getTrackbarPos('L - h', 'panel')
		u_h = cv2.getTrackbarPos('U - h', 'panel')
		l_s = cv2.getTrackbarPos('L - s', 'panel')
		u_s = cv2.getTrackbarPos('U - s', 'panel')
		l_v = cv2.getTrackbarPos('L - v', 'panel')
		u_v = cv2.getTrackbarPos('U - v', 'panel')          
		start = cv2.getTrackbarPos(switch, 'panel')
		bool_motor = cv2.getTrackbarPos(string_motor, 'panel')
		
		lower_green = np.array([l_h, l_s, l_v])
		upper_green = np.array([u_h, u_s, u_v])
	
		cv2.imshow('panel', panel)
		key = cv2.waitKey(20)
		img_deleted_background, mask = image_processor.remove_background_youtube(img_raw, lower_green, upper_green)
		cv2.imshow('bckgrndsgmnttn', img_deleted_background)
		mask = cv2.blur(mask, (3, 3))
		img_gray = image_processor.gray(img_raw)
		
                # Why do I do both of these and combine them?
                mask_canny_edge = image_processor.canny(mask)
		img_canny_edge = image_processor.auto_canny(img_gray)
                mask_weight = 1.0
                combined_canny_edge = image_processor.combine_edges(mask_canny_edge, mask_weight, img_canny_edge (1-mask_weight))
		
		object_contour = image_processor.max_contour(combined_canny_edge)
		try:
			bounding_box = image_processor.bounding_box(object_contour)
		except:
			continue
		cv2.drawContours(img_raw,object_contour,-1,(0,255,0),3)
		#print(bounding_box)
		cv2.rectangle(img_raw, bounding_box[0], bounding_box[1], (0, 255, 0), 2)
		
		cv2.imshow('canny', combined_canny_edge)
		cv2.imshow('raw', img_raw)
		OTL_cut_out = img_deleted_background[bounding_box[0][1]:bounding_box[1][1],bounding_box[0][0]:bounding_box[1][0],:]
		cv2.imshow('cut_out', OTL_cut_out)
		if key == ord('s'):
			cv2.imwrite('backgroundsegmented.jpg', img_deleted_background)
			cv2.imwrite('canny.jpg', combined_canny_edge)
			cv2.imwrite('raw.jpg', img_raw)
			cv2.imwrite('cut_out.jpg', OTL_cut_out)
		if start == 1:
			#todo read classes 
			#classes = readClasses
			break
		if bool_motor == 1:
			motor.forward(delay, 1000)



	## Section for the training data creation
	cv2.destroyAllWindows()

	for i in range(0, STEPS_FOR_FULL_CIRCLE, steps):
		print("Progress for " + label + ": " + str(i/STEPS_FOR_FULL_CIRCLE*100) + "%")
		img_raw = camera.capture()
		
		img_deleted_background, mask = image_processor.remove_background_youtube(img_raw, lower_green, upper_green)
		
		key = cv2.waitKey(20)
		if key == ord('q'):
			cv2.destroyAllWindows()
			motor.cleanUp()
			break
		mask = cv2.blur(mask, (3, 3))
		img_gray = image_processor.gray(img_raw)
		mask_canny_edge = image_processor.canny(mask)
		img_canny_edge = image_processor.auto_canny(img_gray)
		combined_canny_edge = image_processor.combine_edges(mask_canny_edge, img_canny_edge)
		
		object_contour = image_processor.max_contour(combined_canny_edge)
		try:
			bounding_box = image_processor.bounding_box(object_contour)
		except:
			i = i - steps
			continue
		OTL_cut_out = img_deleted_background[bounding_box[0][1]:bounding_box[1][1],bounding_box[0][0]:bounding_box[1][0],:]
		# check if cut_out_size is 2 * lesser/greater than avg img size
		if 0 in last_five_OTL_sizes:
			# add the current OTL cut size
			last_five_OTL_sizes[i%5] = np.shape(OTL_cut_out)[0] * np.shape(OTL_cut_out)[1]
		else:
			# calculate the mean and check whether or not it fits the last 5 elements mean
			current_OTL_size = np.shape(OTL_cut_out)[0] * np.shape(OTL_cut_out)[1]
			mean_OTL_size = np.mean(last_five_OTL_sizes)
			if current_OTL_size < 0.5*mean_OTL_size or 2*mean_OTL_size < current_OTL_size:
				print(" I saved you from wrong OTL cuts!")
				i = i - steps
				continue
			else:
				last_five_OTL_sizes[i%5] = np.shape(OTL_cut_out)[0] * np.shape(OTL_cut_out)[1]
		# Save the snippet of the current OTL
		if not os.path.exists("snippets/" + label):
			os.makedirs("snippets/" + label)
		filename = "snippets/" + label + "/" + str(hashlib.md5(str.encode(str(time.time()))).hexdigest()) + '.jpg'
		
		cv2.imwrite(filename, OTL_cut_out)
		motor.forward(delay, 10)
		
		#except Exception as e:
	#		print("I am a failure " + str(e))
	#		continue

		# SECTION FOR FILE OPERATIONS

		# write bounding box in correct format (see file operations class)
		# write image into correct folder with correct naming
		# 
"""		
		filename = str(hashlib.md5(str.encode(str(time.time()))).hexdigest())
		# folder for the class
		foldername = "new_folder/"
		# datafolder for the images and the txt files
		datafolder = foldername + "data/"
		cwd = os.getcwd()
		newpath = cwd + "/" + datafolder
		if not os.path.exists(newpath):
			os.makedirs(newpath)
		imagefile = datafolder + filename + '.png' 
		textfile = open(datafolder + filename + '.txt', 'a')
		
		file_operations.save_to_folder(textfile, imagefile, bounding_box, final)
"""
		#cv2.imshow('combined', final)
		



multiple_classes = True
cls_ids = [f for f,_ in enumerate(classes)]
if multiple_classes == True:
	for i in range(images):
		snippets = []
		for k in classes:
			pathname = "snippets/" + k + "/"
			filename = random.choice(os.listdir(pathname))
			filepath = pathname + filename
			snippets.append(cv2.imread(filepath))
		final, bounding_box_array = image_processor.multiple_OTL_on_background(snippets, cls_ids, occlusion=False)
		save_to_files(bounding_box_array, final)
		if final is 0 and 0 in bounding_box_array:
			print("Error in OTL_on_background")
			i = i - steps
			continue
else:
	for label in classes:
		for i in range(images):
			final, bounding_box = image_processor.OTL_on_background(OTL_cut_out)
			if final is 0 and bounding_box is 0:
				print("Error in OTL_on_background")
				i = i - steps
				continue


	       
	 
		

		
		# 5 different transformations
		# 0...overlay with some image (transparent overlay)
		# 1...delete and replace the background
		# 2...occlude the object by a more or less random percentage
		# 3...move the object to another (random) position in the image
		# 4...insert another known object into the image and provide the bounding box of that
		#
		# parameters: ("which transformation", "bounding box", "raw image", "folder for overlay/background images")
		#bounding_box_additional_object, img_processed = image_processor.transformation()

		#motor.forward(delay, steps)
		#what to do?
		# 10. capture an image
		# 20. apply contour detection, canny, etc. and find the object and the
		#     bounding box for the object
		# 30. write the text files with the bounding boxes in YOLO format
		# 40. decide the transformation to apply on the image
		# 50. safe the final transformed image
		# 60.

foldername = "new_folder/"
file_operations.write_config_files(classes, foldername)
cv2.destroyAllWindows()
motor.cleanUp()
shutil.make_archive(foldername[:-1], 'zip', foldername[:-1])
print("finished successfully")
