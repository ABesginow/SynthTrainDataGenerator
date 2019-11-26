# TODO imports

# TODO einruecken
import pdb



def iou(bb1, bb2, size=0):
    # TODO convert bounding boxes to [TL, BR] format
    # TODO p1 test!
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
        
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

    if size is not 0:
        # Absolute size of the image
        W = size[1]
        H = size[0]
        # _c = center
        # _r = relative
        x_c_r, y_c_r, w_r, h_r = bb1
        # Get absolute sizes / coordinates of the bounding boxes
        x_c = x_c_r * W
        y_c = y_c_r * H
        w = w_r * W
        h = h_r * H
            
        x1 = int(x_c - w/2)
        x2 = int(x_c + w/2)
        y1 = int(y_c - h/2)
        y2 = int(y_c + h/2)

        bb1 = ((x1, y1), (x2, y2))

        # _c = center
        # _r = relative
        x_c_r, y_c_r, w_r, h_r = bb2
        # Get absolute sizes / coordinates of the bounding boxes
        x_c = x_c_r * W
        y_c = y_c_r * H
        w = w_r * W
        h = h_r * H

        x1 = int(x_c - w/2)
        x2 = int(x_c + w/2)
        y1 = int(y_c - h/2)
        y2 = int(y_c + h/2)
        
        bb2 = ((x1, y1), (x2, y2))

			
	# determine the coordinates of the intersection rectangle
    #pdb.set_trace()
    x_left = max(bb1[0][0], bb2[0][0])
    y_bottom = max(bb1[0][1], bb2[0][1])
    x_right = min(bb1[1][0], bb2[1][0])
    y_top = min(bb1[1][1], bb2[1][1])

	#print(str(x_left) + " " + str(y_top) + " " + str(x_right) + " " + str(y_bottom))
	
    if x_right < x_left or y_top < y_bottom:
        return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box

    #pdb.set_trace()	
    intersection = (x_right - x_left) * (y_top - y_bottom)
    print("Intersection size: " + str(intersection))


    x_left = min(bb1[0][0], bb2[0][0])
    y_bottom = min(bb1[0][1], bb2[0][1])
    x_right = max(bb1[1][0], bb2[1][0])
    y_top = max(bb1[1][1], bb2[1][1])

    union = (x_right- x_left) * (y_top - y_bottom)
    print("Union size: " + str(union))
    print("Image size: " + str(W*H))
    iou_prct = int(intersection / union * 100)
    return iou_prct




H = 1080
W = 1920
size = (H, W, 0)

# Collision case
bb1 = (0.5, 0.5, 0.7, 0.7)
bb2 = (0.6, 0.6, 0.3, 0.3)

print(str(iou(bb1, bb2, size)))

# Non collision case
bb1 = (0.2, 0.2, 0.1, 0.1)
bb2 = (0.8, 0.8, 0.1, 0.1)

print(str(iou(bb1, bb2, size)))
