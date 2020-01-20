import csv
import pdb
import cv2
import numpy as np

def convert(size, box):
    # TODO rewrite box to be [TL, BR] coordinates
    """
    from absolute positions to relative positions, centred around the middle (yolo format)
    Inputs: 
        size = [widht, height]
        box  = [x1, x2, y1, y2]
    """
    #pdb.set_trace()
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






with open('annotations_download_banana.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(reader):
        #pdb.set_trace()
        jpg_file = row[0]
        img = cv2.imread(jpg_file)
        (H, W, _) = np.shape(img)
        box = [int(row[1]), int(row[3]), int(row[2]), int(row[4])]
        (x, y, w, h) = convert((W, H), box)
        txt_file_name = row[0][:-4] + '.txt'
        with open(txt_file_name, 'a+') as txt_file:
            txt_file.write('0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(w)  + ' ' + str(h))
        print("file nr. " + str(i+1) + txt_file_name)
        #print(', '.join(row))





#2 0.375000 0.481250 0.745313 0.406944
#1 0.927734 0.500000 0.144531 0.150000



