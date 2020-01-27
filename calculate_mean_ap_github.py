"""
author: Timothy C. Arlen
date: 28 Feb 2018

Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. Usage:

> python calculate_mean_ap.py

Will display a plot of precision vs recall curves at 10 distinct IoU thresholds as well as output
summary information regarding the average precision and mAP scores.

NOTE: Requires the files `ground_truth_boxes.json` and `predicted_boxes.json` which can be
downloaded fromt this gist.
"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.

    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)

    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: avg precision as well as summary info about the PR curve

        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(
                gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}


def plot_pr_curve(
    precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax


if __name__ == "__main__":

    import pprint as p

    # {[confidence, "TP"/"FP", filename] in a dict given the label}
    # This is done for each 1000 trainings (13) and for each iteration (5), in a dict given the underlying dataset ("banana", "custom")
    AP_List = {
        "banana" : {
            "coco"  : [{"banana": [0.0 for _ in range(5)]} for _ in range(13)],
            "synth" : [{"banana": [0.0 for _ in range(5)]} for _ in range(13)]
            },
        "custom" : {
            "handmade"  : [    
            {   "halterung": [0.0 for _ in range(5)],
                "seitenteil":[0.0 for _ in range(5)],
                "rolle":     [0.0 for _ in range(5)]} for _ in range(13)],
            "synth"     : [    
            {   "halterung": [0.0 for _ in range(5)],
                "seitenteil":[0.0 for _ in range(5)],
                "rolle":     [0.0 for _ in range(5)]} for _ in range(13)],
            "synth_big" : [
            {   "halterung": [0.0 for _ in range(5)],
                "seitenteil":[0.0 for _ in range(5)],
                "rolle":     [0.0 for _ in range(5)]} for _ in range(13)]
            },
        }


    results = {
        "banana" : {
            "coco"  : [[[{"banana": []}, 0, 0] for _ in range(13)] for _ in range(5)],
            "synth" : [[[{"banana": []}, 0, 0] for _ in range(13)] for _ in range(5)]
            },
        "custom" : {
            "handmade"  : [[    
            [{   "halterung": [],
                "seitenteil":[],
                "rolle":     []}, 0, 0] for _ in range(13)] for _ in range(5)],
            "synth"     : [[    
            [{   "halterung": [],
                "seitenteil":[],
                "rolle":     []}, 0, 0] for _ in range(13)] for _ in range(5)],
            "synth_big" : [[    
            [{   "halterung": [],
                "seitenteil":[],
                "rolle":     []}, 0, 0] for _ in range(13)] for _ in range(5)]
            },
        }
    mAP_list = []
    # Calling the results
    #results[cls][train][run][weight[:-3]-1][label][0][{0,1,2}]
    # File operations to load BBs
    listofdirs = os.listdir('./results')
    listofdirs.sort()
    for logfile in listofdirs:
        total_predictions = 0
        TP = 0
        FP = 0
        possible_GTs = 0
        FN = 0
        if not logfile.endswith('.txt'):
            continue
        with open('./results/' + logfile, 'r') as f:
            try:
                temp = json.load(f)
            except:
                print(logfile)
        # charas => Characteristics
        charas = logfile.split('_')
        # replace this with regex search for the 4 or 5 digit number (/d{4,5}
        charas[-1] = charas[-1][:-4]
        if len(charas) == 5:
            charas = [charas[0], charas[1] + '_' + charas[2], charas[3], charas[4]]
        cls = charas[0]
        train = charas[1]
        run = charas[2]
        weight = charas[3]
        
        # Go through all entries ( of the form [[GT], [pred], filename]) and fill the dicts
        for entry in temp:
            gt = entry[0]
            pred = entry[1]
            imgname = entry[2]
            #if 'banana' in cls and 'coco' in train and '1' in run and '8000' in weight:

            # Some string replacements for easier comparison
            if 'banana' in cls:
                for g in gt:
                    g[0] = "banana"
            if 'custom' in cls:
                for g in gt:
                    if g[0] is '0':
                        g[0] = "halterung"
                    elif g[0] is '1':
                        g[0] = "seitenteil"
                    elif g[0] is '2':
                        g[0] = "rolle"
            # in "handmade" the .names file was in another order than the synth file.
            # Therefore seitenteil = rolle and rolle = seitenteil
            if 'custom' in cls:
                if 'handmade' in train:
                    for p in pred:
                        if 'seitenteil' in p[0]:
                            p[0] = 'rolle'
                        elif 'rolle' in p[0]:
                            p[0] = 'seitenteil'
            possible_GTs += len(gt)
            # Now calculate IoUs
            #[xmin, ymin, xmax, ymax] -> structure for boxes
            for ipb, (pred_label, pred_conf, pred_box) in enumerate(pred):
                correct = False
                for igb, (gt_label, gt_box) in enumerate(gt):
                    if pred_label.lower() in gt_label.lower():
                        
                        # Calling the results (0 = conf, 1 = ("TP"/"FP"), 2 = filename)
                        #results[cls][train][run][weight[:-3]-1][label][{0,1,2}]

                        # They are the same label, they can potentially have an IoU > 0.5
#                        pdb.set_trace()
                        #[xmin, ymin, xmax, ymax]
                        x_cen, y_cen, x_len, y_len = pred_box
                        pred_box = [int(x_cen - 1/2*(x_len)),
                                        int(y_cen - 1/2*(y_len)),
                                        int(x_cen + 1/2*(x_len)),
                                        int(y_cen + 1/2*(y_len))]
                        x_cen, y_cen, x_len, y_len = gt_box
                        gt_box = [int(x_cen - 1/2*(x_len)),
                                        int(y_cen - 1/2*(y_len)),
                                        int(x_cen + 1/2*(x_len)),
                                        int(y_cen + 1/2*(y_len))]
                        try:                                    
                            iou = calc_iou_individual(pred_box, gt_box)
                        except:
                            iou = 0

                        if iou > 0.5:
                            # IoU > 0.5 -> A TP!
                            correct = True
                            results[cls][train][int(run)-1][int(weight[:-3])-1][0][pred_label.lower()].append([float(pred_conf), "TP", imgname])
                            TP = TP + 1
                            break
                        # They are not the same label, continue to look through the test
                        # If by this point no gt matching the pred was found this is a FP occurence
                if not correct:
                    FP = FP + 1
                    results[cls][train][int(run)-1][int(weight[:-3])-1][0][pred_label.lower()].append([float(pred_conf), "FP", imgname])
                # And in the end I can calculate the FN 
                total_predictions += len(pred)
        import pprint
        if not possible_GTs == 229 and not possible_GTs == 96:
            print("LOG: " + str(logfile))
        results[cls][train][int(run)-1][int(weight[:-3])-1][1] = possible_GTs
        #print("Total number of predictions in " + logfile + ': ' + str(total_predictions))



    # Now I can go over all pred_label and calculate the AP

    # Calculate mAP (copied from method)
    #results[cls][train][run][weight[:-3]-1][label][0][{0,1,2}]
    mAPs = []
    for cls in results:
        for train in results[cls]:
            for run in range(len(results[cls][train])):
                for weight in range(len(results[cls][train][run])):
                    APs = [0.0 for _ in range(len(results[cls][train][run][weight][0]))]
                    for j, label in enumerate(results[cls][train][run][weight][0]):
                        TP = 0
                        FP = 0
#                        pdb.set_trace()
                        # p_inter has to be the sum of all TP and all FP!
                        p_inter = [0.0 for _ in range(len(results[cls][train][run][weight][0][label]) + 1)]

                        precisions = np.array([0.0 for _ in range(len(results[cls][train][run][weight][0][label]) + 1)])
                        precisions[0] = 1
                        recalls = np.array([0.0 for _ in range(len(results[cls][train][run][weight][0][label]) + 1)])
                        recalls[0] = 0
                        possible_GTs = results[cls][train][run][weight][1]
                        sorted_results = np.sort(results[cls][train][int(run)][weight][0][label.lower()], axis=0)[::-1]
                        for i, entry in enumerate(sorted_results):
                            if 'FP' in entry[1]:
                                FP += 1
                            elif 'TP' in entry[1]:
                                TP += 1
                            if (TP + FP) == 0:
                                if i == 0:
                                    precisions[0] = 0
                                precisions[i+1] = 0
                            else:
                                if i == 0:
                                    precisions[0] = TP / (TP + FP) 
                                precisions[i+1] = TP / (TP + FP)
                            
                            recalls[i+1] = TP / possible_GTs
                        p_inter = [max(precisions[indice:]) for indice in range(len(precisions))]
                        if len(p_inter) == 0:
                            APs[j] = 0
                        else:
                            APs[j] = np.trapz(p_inter, x=recalls)
                            # Alternative method with the same result as np.trapz
                            #APs[j] = np.sum([(recalls[i+1] - recalls[i]) * p_inter[i] for i in range(len(recalls) -1 )])
                        #print(APs[j])
                        # Can be used to plot the Precision-Recall curves
                        """
                        fig = plt.figure()
                        if cls == 'banana' and train == 'synth' and run == 0 and weight == 8:
                            pprint.pprint(sorted_results)
                            pprint.pprint(p_inter)
                            pprint.pprint(recalls)
                            plt.plot(recalls, p_inter)
                            plt.set_ylim=(0, 1.1)
                            plt.title('_'.join([cls, train, str(run), str(weight)]))
                            plt.show()
                        """
                        AP_List[cls][train][weight][label][run] = APs[j]
                            # DEBUG: Was just used to see if there was actually a difference between the interpolated curve and the original precision curve
                            #if not np.trapz(precisions, x=recalls) == APs[j]:
                            #    pdb.set_trace()
                    mAP = np.mean(APs)
                    mAPs.append(mAP)
                    results[cls][train][run][weight][2] = APs
                    #print(cls + ' ' + train + ' ' + str(weight) + ' ' + str(mAP))
                    #print("TP: " + str(TP) + "; FP: " + str(FP) + "; GT: " + str(possible_GTs))
#pprint.pprint(AP_List)
#print(np.mean(mAPs))
#print(min(mAPs))
#print(max(mAPs))
#print(np.mean(mAPs/max(mAPs)))


# x axis shows the iterations from 0 to 13000
# Error bars every 500 or 1000 iterations to prevent overcrowded graph
x = np.arange(1, 14)
#colors= ['orange', 'darkorange', 'purple', 'blue', 'darkblue']
# 
#fig, ax = plt.subplots(5)
fig = plt.figure()
y = [0 for _ in range(5)]
yerr_min = [0 for _ in range(5)]
yerr_max = [0 for _ in range(5)]
yerr = [0 for _ in range(5)]
runner_variable  = 0
legend = ['' for _ in range(5)]
for cls in AP_List:
    for train in AP_List[cls]:
        curve_label = '_'.join([cls, train])
# Make mAP List with 13 * 5 entries here?
        mAP = [[0.0]*5]*13
        #pdb.set_trace()
        for weight in range(len(AP_List[cls][train])):
            for label in AP_List[cls][train][weight]:
                if not 'banana' in cls:
                    mAP[weight] = np.round(np.mean(list(AP_List[cls][train][weight].values()), axis=1), 3)
                else:
                    mAP[weight] = np.round(list(AP_List[cls][train][weight][label]),3)
        print("mean mAP over all train runs:")
        pprint.pprint(np.mean(mAP, axis=1))
        print("min and max at each position of ecah run")
        pprint.pprint(np.amin(mAP, axis=1))
        pprint.pprint(np.amax(mAP, axis=1))
        y[runner_variable] = np.mean(mAP, axis = 1)
        yerr_min[runner_variable] = np.amin(mAP, axis = 1)
        yerr_max[runner_variable] = np.amax(mAP, axis = 1)
        yerr[runner_variable] = [yerr_min[runner_variable], yerr_max[runner_variable]]
        #pdb.set_trace()
        plt.axes(ylim=(0, 0.2))
        plt.errorbar(x, y[runner_variable], yerr=yerr[runner_variable], label=curve_label, capsize=3)#, c=colors[runner_variable])#, fmt='--o', elinewidth=runner_variable, alpha=(1/(runner_variable+1)))
        runner_variable += 1
    plt.legend(loc='upper left')
    plt.xlabel('Training iterations (x1000)')
    plt.ylabel('mean average precision')
    plt.show()
pdb.set_trace()

pprint.pprint(y[2])
pprint.pprint(y[3])
pprint.pprint(y[4])


# yerr = [[lower_error], [upper_error]] for lower and upper bound errors
# Left from MWE for reference
#yerr = [np.random.random_sample(10), np.random.random_sample(10)]


#ax.errorbar(x, y, yerr=yerr)#,  fmt='--o')
#ax.set_xlabel('Iteration (x1000)')
#ax.set_ylabel('mAP')
#ax.set_title('mean Average Precision curves over ')




# NOTE: Somehow the confidence for a given prediction is important to calculate the mAP
# TODO: For each recall level, take the maximum precision level (Interpolated precision)



#"""
