#!/usr/bin/python3
"""
Evaluates network's label predictions. 
Test file and predicted file should be in the same directory.

usage: python evaluateDetector.py test 100000
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import sys

basedir = "caffeBlobs"
blobdir = sys.argv[1]    # should be in caffeBlobs
iter = sys.argv[2]    # only predicted files which have this iteration number in their file name will be evaluated

tol = 10    # Tolerance radius in pixels for label

predictedFiles = [f for f in os.listdir(blobdir) if "_predicted_"+iter+".h5" in f]

nTP = 0
nFP = 0
nFN = 0
SSD = 0
nCHs = 0    # number of convex hulls
total_ch_error = 0     # sum of noramlized difference in CH values

for predictedFile in predictedFiles:

    basename = predictedFile[0:-20]

    # Load ground truth label, ignore masks and convex hulls
    gtFile = h5py.File("{}/{}.h5".format(blobdir, basename), 'r')
    gtLabels = np.empty(gtFile["/label"].shape, dtype=np.float32)
    gtLabels[:] = gtFile["/label"][:]
    gtWeights = np.empty(gtFile["/label_weight"].shape, dtype=np.float32)
    gtWeights[:] = gtFile["/label_weight"][:]
    ignore = gtWeights == 0
    gtWeights = None
    gtCHs = np.empty(gtFile["/label_ch_1"].shape, dtype=np.float32)
    gtCHs[:] = gtFile["/label_ch_1"][:]
    gtFile.close()

    # Load predicted labels (softmax already computed in network)
    predFile = h5py.File("{}/{}_predicted_{}.h5".format(blobdir, basename, iter), "r")
    predLabels = np.empty(predFile["/prediction_label"].shape, dtype=np.float32)
    predLabels[:] = predFile["/prediction_label"][:]
    predCHs = np.empty(predFile["/prediction_ch_1"].shape, dtype=np.float32)
    predCHs[:] = predFile["/prediction_ch_1"][:]


    ### Evaluate Labels ##
    # find connected components and their positions in ground truth labels
    gtSlice = gtLabels[0,0,:,:]
    (conncomps, gtNPepLabels) = ndimage.measurements.label(gtSlice > 0.5)
    gtLabelPositions = ndimage.measurements.center_of_mass(
        gtSlice, conncomps, np.arange(1, gtNPepLabels + 1))

    # find connected components and their positions in predicted label    
    predSlice = predLabels[0,1,:,:]
    predSlice[predSlice >= 0.5] = 1
    predSlice[predSlice < 0.5] = 0
    dilatedSlice = ndimage.binary_dilation(predSlice, iterations=1).astype(np.float32)    # dilate predicted label
    (conncomps, predNPepLabels) = ndimage.measurements.label(dilatedSlice)    # find connected component
    predLabelPositions = ndimage.measurements.center_of_mass(dilatedSlice,    # find center of mass for every connected component
                                                             conncomps,
                                                             np.arange(1, predNPepLabels + 1))

    tp = 0
    fp = 0
    se = 0
    for pos in predLabelPositions:
        if ignore[0,0,int(pos[0]),int(pos[1])]:
            continue
        if gtLabels[0,0,int(pos[0]),int(pos[1])] == 0 or len(gtLabelPositions) == 0:
            fp = fp + 1
            continue

        # finding the nearest neighbour connected components
        minSqDist = np.sum((np.array(gtLabelPositions[0]) - np.array(pos)) * (np.array(gtLabelPositions[0]) - np.array(pos)))
        minIdx = 0
        for gtPosIdx in range(1, len(gtLabelPositions)):
            sqDist = np.sum((np.array(gtLabelPositions[gtPosIdx]) - np.array(pos)) * (np.array(gtLabelPositions[gtPosIdx]) - np.array(pos)))
            if sqDist < minSqDist:
                minSqDist = sqDist
                minIdx = gtPosIdx
        if minSqDist < tol * tol:
            tp = tp + 1
            se = se + minSqDist
            gtLabelPositions.pop(minIdx)
        else:
            fp = fp + 1
    nTP = nTP + tp
    nFP = nFP + fp
    nFN = nFN + len(gtLabelPositions)
    SSD = SSD + se
    if tp > 0:
        rmse = math.sqrt(se / tp)
    else:
        rmse = 0
    print("  {} Label: {} / {} correctly identified (rmse = {:.2f}), {} false positives, {} missed".format(basename, tp, gtNPepLabels, rmse, fp, gtNPepLabels - tp))


print("\nEvaluation for Labels:")
print("    Recall: {} / {} ({:.2f}%)".format(nTP, nTP + nFN, 100 * nTP / (nTP + nFN)))
print("    Precision: {} / {} ({:.2f}%)".format(nTP, nTP + nFP, 100 * nTP / (nTP + nFP)))
print("    RMSE = {} px".format(math.sqrt(SSD / nTP)))

