from image_capture import ImageCapture
from motor_control_PI import MotorControl
from image_processing import ImageProcessing
from file_operations import FileOperations
import cv2
import numpy as np
import hashlib
import time
import os
import random
import argparse
# zip
import shutil
from pathlib import Path

# load the config file to get all the information
from configparser import ConfigParser

RPI_CAMERA = 0


def getsnippets(classes):
    snippets = []
    for c in classes:
        pathname = snippets_folder + c + "/"
        filename = random.choice(os.listdir(pathname))
        filepath = pathname + filename
        snippets.append(cv2.imread(filepath))
    return snippets


def nothing(x):
    pass

def save_to_files(bounding_box, final, foldername):
    filename = str(hashlib.md5(str.encode(str(time.time()))).hexdigest())
    # datafolder for the images and the txt files
    datafolder = foldername + "data/"
    cwd = os.getcwd()
    newpath = cwd + "/" + datafolder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    imagefile = datafolder + filename + '.png'
    textfile = open(datafolder + filename + '.txt', 'a')

    file_operations.save_to_folder(textfile, imagefile, bounding_box, final)

def create_config_file():
    #Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
    config_object["GENERAL"] = {
        "steps for full rotation": "12360",
        "steps per image": "30",
        "images per class": "3000",
        "delay": "0.001",
        "results folder":"Results/",
        "snippets folder": "Snippets/",
        "backgrounds folder":"Backgrounds/"
    }
    config_object["DATASETINFO"] = {
            "object0": "hose",
            "object1": "nut",
            "object2": "connector",
            "object3": "shower head",
            "object4": "bag",
            "object5": "paper"
    }

    #Write the above sections to config.ini file
    with open('config.ini', 'w') as conf:
            config_object.write(conf)

def execute_processing(img_raw, lower_green, upper_green):
    # cut out background
    img_deleted_background, mask = image_processor.remove_background_youtube(img_raw, lower_green, upper_green)
    mask = cv2.blur(mask, (3, 3))
    img_gray = image_processor.gray(img_raw)
    # For possible future purposes
    mask_canny_edge = image_processor.canny(mask)
    img_canny_edge = image_processor.auto_canny(img_gray)
    mask_weight = 1.0
    combined_canny_edge = image_processor.combine_edges(mask_canny_edge,
            mask_weight, img_canny_edge, (1-mask_weight))

    object_contour = image_processor.max_contour(combined_canny_edge)

    try:
        bounding_box = image_processor.bounding_box(object_contour)
    except:
        return 0, 0, 0

    OTL_cut_out = img_deleted_background[bounding_box[0][1]:bounding_box[1][1],bounding_box[0][0]:bounding_box[1][0],:]

    return img_deleted_background, combined_canny_edge, OTL_cut_out

def create_ui():
    # Initialise GUI
    panel = np.zeros([100, 700], np.uint8)
    cv2.namedWindow('panel')

    # 056
    cv2.createTrackbar('L - h', 'panel', 0, 255, nothing)
    # 069
    cv2.createTrackbar('U - h', 'panel', 255, 255, nothing)

    # 087
    cv2.createTrackbar('L - s', 'panel', 0, 255, nothing)
    # 255
    cv2.createTrackbar('U - s', 'panel', 255, 255, nothing)

    # 0
    cv2.createTrackbar('L - v', 'panel', 0, 255, nothing)
    # 255
    cv2.createTrackbar('U - v', 'panel', 255, 255, nothing)

    # Cut left side of image
    cv2.createTrackbar('left', 'panel', 0, 255, nothing)
    # Cut right side of image
    cv2.createTrackbar('right', 'panel', 1, 255, nothing)
    # Cut top side of image
    cv2.createTrackbar('top', 'panel', 0, 255, nothing)
    # Cut bot side of image
    cv2.createTrackbar('bot', 'panel', 1, 255, nothing)

    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'panel',0,1,nothing)
    string_motor = '0 : MOTOR OFF \n1 : MOTOR ON'
    cv2.createTrackbar(string_motor, 'panel', 0, 1, nothing)



#Get the configparser object
config_object = ConfigParser()

cfg_file = Path("config.ini")

if not cfg_file.exists():
    create_config_file()

#Read config.ini file
config_object.read("config.ini")

general_info = config_object["GENERAL"]
dataset_info = config_object["DATASETINFO"]
#multivariants = config_object["MULTIVARIANT"]
instances = config_object["INSTANCES"]


classes = {}
# Getting the entries
for i, entry in enumerate(dataset_info):
    classes[entry] = {instances[dataset_info[entry]]} if dataset_info[entry] in instances else {"variants" : 1}
#    if dataset_info[entry] in instances:
#        classes["object"+i] = {"label" : dataset_info[entry],
#                               "variants": instances[dataset_info[entry]]}
#    else:
#        classes["object"+i] = {"label": dataset_info[entry],
#                               "variants": 1}
#    classes.append(dataset_info[entry])



IMAGES = int(general_info["images per class"])
STEPS_FOR_FULL_CIRCLE = int(general_info["steps for full rotation"])
DELAY = float(general_info["delay"])
steps =int(general_info["steps per image"])
results_folder = general_info["results folder"]
snippets_folder = general_info["snippets folder"]

file_operations = FileOperations()
motor = MotorControl()
camera = ImageCapture(RPI_CAMERA)
image_processor = ImageProcessing()

parser = argparse.ArgumentParser(description='Create a synthetic dataset for object detection.')
parser.add_argument('--only_snippets', action="store_true", help='only capture snippets without creating a dataset (default: false)')
parser.add_argument('--only_train_images', action="store_true", help='only create the dataset wihout capturing snippets (default: false)')
parser.add_argument('--multiclass_images', action="store_false", help='always draw objects from all classes concurrently on the training images (default: True)')
parser.add_argument('--randomize_multiclass', action="store_true", help='randomly sample classes which are drawn on the training images (default: false)')
parser.add_argument('--allow_overlap', action="store_false", help='allows objects placed in the training images to overlap at most 50% (default: true)')

args = parser.parse_args()

only_snippets =args.only_snippets
only_train_images = args.only_train_images
multiclass = args.multiclass_images
randomize_multiclass = args.randomize_multiclass
allow_overlap = args.allow_overlap

## Section for the configuration
# Make images for every class
for label in classes:
    if only_train_images:
        break
    for variant in range(classes["label"]):
        print("variant {} of {}".format(variant, classes["label"]))
        last_five_OTL_sizes = [0]*5

        create_ui()

        # Allow user to set boundary values & get live preview
        while True:
            img_raw = camera.capture()

            l_h = cv2.getTrackbarPos('L - h', 'panel')
            u_h = cv2.getTrackbarPos('U - h', 'panel')
            l_s = cv2.getTrackbarPos('L - s', 'panel')
            u_s = cv2.getTrackbarPos('U - s', 'panel')
            l_v = cv2.getTrackbarPos('L - v', 'panel')
            u_v = cv2.getTrackbarPos('U - v', 'panel')
            l_cut = cv2.getTrackbarPos('left', 'panel')
            r_cut = cv2.getTrackbarPos('right', 'panel')
            t_cut = cv2.getTrackbarPos('top', 'panel')
            b_cut = cv2.getTrackbarPos('bot', 'panel')
            start = cv2.getTrackbarPos(switch, 'panel')
            bool_motor = cv2.getTrackbarPos(string_motor, 'panel')

            lower_green = np.array([l_h, l_s, l_v])
            upper_green = np.array([u_h, u_s, u_v])
                # for live preview
            cv2.imshow('panel', panel)
            key = cv2.waitKey(20)
            img_raw = img_raw[0+t_cut:-b_cut, 0+l_cut:-r_cut]
            # Function that returns the results of the different processing steps
            img_deleted_background, canny_edge, OTL_cut_out = execute_processing(img_raw, lower_green, upper_green)
            if img_deleted_background == canny_edge == OTL_cut_out:
                continue

            cv2.imshow('bckgrndsgmnttn', img_deleted_background)

           # display stuff
            cv2.imshow('canny', canny_edge)
            cv2.imshow('raw', img_raw)
            cv2.imshow('cut_out', OTL_cut_out)
            # Possibility to save images
            if key == ord('s'):
                cv2.imwrite('backgroundsegmented.jpg', img_deleted_background)
                cv2.imwrite('canny.jpg', canny_edge)
                cv2.imwrite('raw.jpg', img_raw)
                cv2.imwrite('cut_out.jpg', OTL_cut_out)
            if key == ord('q'):
                cv2.destroyAllWindows()
                motor.cleanUp()
                quit()
            if start == 1:
                break
            if bool_motor == 1:
                motor.forward(DELAY, 1000)


        ## Section for the training data creation
        cv2.destroyAllWindows()

        for i in range(0, STEPS_FOR_FULL_CIRCLE, steps):
            #print(f"Progress for {label}: {str(i/STEPS_FOR_FULL_CIRCLE*100)}%")
            print("Progress for {}: {}%".format(label, str(i/STEPS_FOR_FULL_CIRCLE*100)))
            img_raw = camera.capture()

            img_deleted_backgroud, canny_edge, OTL_cut_out = execute_processing(img_raw, lower_green, upper_green)

            # check if cut_out_size is 2 * lesser/greater than avg img size to prevent random outer edges to be recognized
            # as an OTL
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
            if not os.path.exists(snippets_folder + label):
                os.makedirs(snippets_folder + label)
            filename = snippets_folder + label + "/" + str(hashlib.md5(str.encode(str(time.time()))).hexdigest()) + '.jpg'

            cv2.imwrite(filename, OTL_cut_out)
            motor.forward(DELAY, steps)


        print("Progress for {}: 100%".format(label))


if only_snippets:
    motor.cleanUp()
    print('finished creating snippets')
    exit()



#only_snippets
#only_train_images
#multiclass
#randomize_multiclass
#allow_overlap



cls_ids = [f for f,_ in enumerate(classes)]

for i in range(images):
    snippets = getsnippets(c)
    background_name = 'Backgrounds/' + random.choice(os.listdir("Backgrounds"))
    background = cv2.imread(background_name)
    try:
        background = imutils.resize(background, width=1000)
    except:
        print("There is something wrong with: " + str(background_name))
    if multiclass:
        final, bounding_box = image_processor.multiple_OTL_on_background(snippets, cls_ids, occlusion=allow_overlap, randomize=randomize_multiclass)
        if final is 0 and 0 in bounding_box_array:
            print("Error in OTL_on_background")
            i = i - steps
            continue
        save_to_files(bounding_box, final, results_folder)
    # TODO can I get this in a single function call w. parameter 'multiclass=True/False'?
    else:
        for snippet in snippets:
            final, bounding_box = image_processor.OTL_on_background(snippet, background=background)
            if final is 0 and 0 in bounding_box_array:
                print("Error in OTL_on_background")
                i = i - steps
                continue
            save_to_files(bounding_box, final, results_folder)


file_operations.write_config_files(classes, results_folder)
cv2.destroyAllWindows()
motor.cleanUp()
print("Creating ZIP file")
shutil.make_archive(results_folder[:-1], 'zip', results_folder[:-1])
print("finished successfully")
