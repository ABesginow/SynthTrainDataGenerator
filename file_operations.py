import os
import atexit
import cv2
import numpy as np
import shutil

class FileOperations():

    
    def __init__(self):
        pass

    def write_box_coordinates(self, txtfile, bb, cls_id):
        """
        Internal function
        Write the bounding box coordinates to the textfile with the given class-id
        """
        txtfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        
    def save_to_folder(self, textfile, imagefile, bb_array, image):
        """
        Write all bounding boxes and a given image into a directory
        """
        cv2.imwrite(imagefile, image)
        # If the shape is not (4,) it's of bigger size and therefore many classes/objects
        if np.shape(bb_array) == (4,):
            print("np.shape is (4,)")
            self.write_box_coordinates(textfile, bb_array, 0)
        else:
            for bb in bb_array:
                self.write_box_coordinates(textfile, bb[1], bb[0])
        textfile.close()


    def write_config_files(self, classes, foldername):
        """
        Create the configuration files for the training, given a certain directory
        """
        # obj.names
        with open(foldername + "obj.names", 'w') as fp:
            for label in classes:
                fp.write(str(label) + "\n")
                
        # obj.data
        with open(foldername + "obj.data", 'w') as fp:
            fp.write( "classes=" + str(len(classes)) + "\n")
            fp.write( "train=" + foldername + "train.txt" + "\n")
            fp.write( "valid=" + foldername + "test.txt" + "\n")
            fp.write( "names=obj.names"+ "\n")
            fp.write( "backup=backup/" + "\n")
        
        # <training_name>_yolo.cfg
        shutil.copy("yolov3.cfg", foldername + foldername[:-1] + "_yolov3.cfg")
        with open(foldername + foldername[:-1] + "_yolov3.cfg", 'r') as f:
            lines = f.readlines()
            
        with open(foldername + foldername[:-1] + "_yolov3.cfg", 'w') as fp:
            for i, line in enumerate(lines):
                if i == 602:
                    line = "filters=" + str((len(classes) + 5)*3) + "\n"
                elif i == 609:
                    line = "classes=" + str(len(classes)) + "\n"
                elif i == 688:
                    line = "filters=" + str((len(classes) + 5)*3) + "\n"
                elif i == 695:
                    line = "classes=" + str(len(classes)) + "\n"
                elif i == 775:
                    line = "filters=" + str((len(classes) + 5)*3) + "\n"
                elif i == 782:
                    line = "classes=" + str(len(classes)) + "\n"
                fp.write(line)
        
# (0 - classes-1)
# class-id + " " + x + " " + y + " " + width + " " + " " + height
