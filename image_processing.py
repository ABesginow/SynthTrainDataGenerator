import cv2
import numpy as np
import argparse
import glob
import imutils
import pdb
import doctest
import random

class ImageProcessing:
    #def __init__(self, img=0):
    #    if not np.all(img == 0):
    #        self.background = img.copy()
    #    pass

    def convert(self, size, box):
        """
        Convert from absolute positions to relative positions, centred around the middle (yolo format)

        Inputs:
            size = [widht, height]
            box  = [(x1, y1), (x2, y2)]
        """
        TL = box[0]
        BR = box[1]

        x_center = (TL[0] + BR[0])/2
        y_center = (TL[1] + BR[1])/2
        width = BR[0] - TL[0]
        height = BR[1] - TL[1]

        dw = 1./size[0]
        dh = 1./size[1]
        x_relative = x_center*dw
        w_relative = width*dw
        y_relative = y_center*dh
        h_relative = height*dh
        return (x_relative, y_relative, w_relative, h_relative)

    def auto_canny(self, image, sigma=0.53):
        """
        Calculates the edges of the image based on an automatic canny edge detection
        Taken from rosebrock-blog (LearnOpenCV?)
        """
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
        """
        Cuts out triangles starting from "percentage*img.size"
        """
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
        """
        Based on some Youtube code, this can create a mask, based on upper and lower boundaries
        """
        hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)

        #lower_green = np.array([56, 87, 0])
    #upper_green = np.array([69, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)

        fg = cv2.bitwise_and(img_raw, img_raw, mask=mask_inv)
        return fg, mask_inv



    def remove_background_histogram(self, img_raw):
        # TODO Vote for removal
        """DEPRECATED
        Based on the histogram of the image, remove the parts of the image which are occuring the most/least (?)
        Code is not really clean aswell
        """
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
        """
        Run a canny edge detection with predefined boundaries and dilate the result so the edges are visible more clearly.
        Also runs on color and grayscale.
        """
        upper = 200
        lower = 0
        img = cv2.blur(img, (2, 2))
        # If image has 3 dimensions (is color image), first convert it to grayscale
        if np.shape(img)[-1] != 3:
            edges = cv2.Canny(img, lower, upper)
        else:
            edges = cv2.Canny(self.gray(img), lower, upper)
        edges = cv2.dilate(edges, (3, 3), iterations=3)
        return edges

    def central_contour(self, img, min_size):
        """
        Find the most central contour in an image, given a minimal size boundary
        img : Result from a canny edge detection (or at least grayscale)
        """
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centerY = np.shape(img)[0]/2
        centerX = np.shape(img)[1]/2
        central_contour = 0
        bestX, bestY = 0, 0
        for c in contours:
            if cv2.contourArea(c) < min_size:
                continue
        # compute the center of the contour
                # TODO research what a moment is exactly
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                cX = 1000
                cY = 1000
                    # if cartesian distance from center is less -> new best(most central) contour found
            if np.sqrt((centerY-cY)**2 + (centerX-cX)**2) < np.sqrt((centerY-bestY)**2 + (centerX-bestX)**2):
                central_contour = c
                bestX = cX
                bestY = cY
        return central_contour


    def max_contour(self, img):
        """
        Find and return the largest contour

        img : Result from a canny edge detection (or at least grayscale)

        """
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

    def combine_edges(self, mask_edge, mask_weight , object_edge, object_weight):
        """
        Idea: Combine both masks using a weighted addition

        """
        dst = cv2.addWeighted(mask_edge, mask_weight, object_edge, object_weight, 0)
        return dst


    @staticmethod
    def iou(bb1, bb2, size=0):
        # TODO make doctests
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        >>> ImageProcessing.iou(((0,0),(100,100)), ((120,120),(150,150)))
        0.0
        >>> ImageProcessing.iou(((0,0), (100, 100)), ((50, 50), (100, 100)))
        25

        size = [H, W, D]

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

        # TODO: automatically recognize yolo format or TL/BR format

        # Using TL/BR format:
        print(bb1)
        print(bb2)
        # Union coordinates
        TL1 = bb1[0]
        BR1 = bb1[1]
        TL2 = bb2[0]
        BR2 = bb2[1]
        x_left_u   = min(TL1[0], TL2[0])
        x_right_u  = max(BR1[0], BR2[0])
        y_top_u    = min(TL1[1], TL2[1])
        y_bottom_u = max(BR1[1], BR2[1])

        # Intersection coordinates
        x_left_i   = max(TL1[0], TL2[0])
        x_right_i  = min(BR1[0], BR2[0])
        y_top_i    = max(TL1[1], TL2[1])
        y_bottom_i = min(BR1[1], BR2[1])

        if x_right_i < x_left_i or y_top_i > y_bottom_i:
            return 0.0


        union = (x_right_u - x_left_u) * (y_bottom_u - y_top_u )
        intersection = (x_right_i - x_left_i) * (y_bottom_i - y_top_i)

        iou_prct = int(intersection / union * 100)
        return iou_prct





        # start coordinates are the TL corner
    def put_snippet_on_background(self, snippet, background, start):
        height_snip, width_snip, _ = np.shape(snippet)
        start_x = start[0]
        start_y = start[1]
        roi = background[start_y:start_y + height_snip, start_x:start_x + width_snip]
        # Now create a mask of snippet and create its inverse mask also
        img2gray = cv2.cvtColor(snippet,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of snippet in ROI
        try:
            background_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        except:
            pdb.set_trace()

        # Take only region of snippet from snippet image
        snippet_fg = cv2.bitwise_and(snippet,snippet,mask = mask)

        # Put snippet in ROI and modify the main image
        dst = cv2.add(background_bg,snippet_fg)

        background[start_y:start_y + height_snip, start_x:start_x + width_snip] = dst

        return background


    def calculate_size(self, snippet, background, random_size, random_position, classes=1):
        offset_y, offset_x = 0, 0
        height_snip, width_snip, _ = np.shape(snippet)
        height_bckg, width_bckg, _ = np.shape(background)
        if random_size:
            max_height, max_width = height_bckg/classes, width_bckg/classes
            # Purpose is to limit the snippet sizes to make it easier for items to fit into the image
            y_scale, x_scale = (random.uniform(0.5, min(2.0, max_height/height_snip)), random.uniform(0.5, min(2.0, max_width/width_snip) ))
            # Used to choose the minimal scale, else they'd exceed the max_height/width
            scale = min(y_scale, x_scale)
            height_snip, width_snip, _ = np.shape(snippet)
            size = (int(width_snip*scale), int(height_snip*scale))
            print(size)
            snippet=cv2.resize(snippet, size)
        if random_position:
            height_snip, width_snip, _ = np.shape(snippet)
            offset_y, offset_x = (random.randrange(0, (height_bckg - height_snip)), random.randrange(0, (width_bckg - width_snip)))
            if offset_y + height_snip > height_bckg or offset_x + width_snip > width_bckg:
                pdb.set_trace()
        return ((offset_x, offset_y),(offset_x+width_snip, offset_y+height_snip))

    def check_for_collisions(self, b, bounding_boxes, threshold=0):
        # check every existing bounding box
        for box in bounding_boxes:
            # calculate the overlap
            if self.iou(b, box[0]) > threshold:
                return False
        return True

    def OTL_on_background(self, snippets, background, cls_to_id, occlusion=True, randomize=False, random_position=True, random_size=True):
        import os, random
        bounding_boxes = []
        # first: theorethically position all the snippets to check for collisions
        for snippet in snippets:
            i = 0
            # Do-while loop, checking for any occlusion (or occlusion up to 50%)
            #((offset_x, offset_y),(offset_x+width_snip, offset_y+height_snip)) #(TL), (BR) - Format
            b = self.calculate_size(snippet[0], background, random_size, random_position, classes=len(snippets))
            print("Bounding box: {}".format(b))
            if not occlusion:
                while not self.check_for_collisions(b, bounding_boxes, threshold=0.0):
                    b = self.calculate_size(snippet[0], background, random_size, random_position, classes=len(snippets) )
                    i += 1
                    if i > 50:
                        print("ERROR: It didn't fit!")
                        return 0, 0
            else:
                while not self.check_for_collisions(b, bounding_boxes, threshold=0.5):
                    b = self.calculate_size(snippet[0], background, random_size, random_position, classes=len(snippets))
                    i += 1
                    if i > 50:
                        print("ERROR: It didn't fit!")
                        return 0, 0
            bounding_boxes.append([b, snippet[1]])

        # after all snippets have been checked, start placing the snippets on the image
        for (snippet, [bb, cls]) in zip(snippets, bounding_boxes):
            offset_x = bb[0][0]
            offset_y = bb[0][1]
            height = bb[1][1] - bb[0][1]
            width = bb[1][0] - bb[0][0]
            snippet[0] = cv2.resize(snippet[0], (width, height))
            print("BB and snippet shape. After resize")
            print(bb)
            print(np.shape(snippet[0]))
            if np.shape(snippet[0])[:-1] == (height, width):
                bckg_height, bckg_width, _ = np.shape(background)
                if offset_x + width > bckg_width or offset_y + height > bckg_height:
                    pdb.set_trace()
                background = self.put_snippet_on_background(snippet[0], background, (offset_x, offset_y))
            else:
                pdb.set_trace()
        return background, bounding_boxes


