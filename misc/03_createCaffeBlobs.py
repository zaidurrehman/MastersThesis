"""Python script which generates caffe blobs from HDF5 images.

use: python 03_createCaffeBlobs

datadir: directory containing HDF5 image files
blobdir: directory where caffe blob files will be saved."""

import numpy as np
from scipy import ndimage
import h5py
import os
import re
import math
from multiprocessing import Pool

n_processes = 8 

# basedir = "."
datadir = "/misc/lmbraid19/urrehmaz/data_complete/data0212/data"
blobdir = "/misc/lmbraid19/urrehmaz/data_complete/data0212/caffeBlobs"
# subdirs = ["B01_ZP-C1", "B07_ZP_C1"]
max_int = {"BM3366":3048833024,
           "BM3368":4297689088,
	   "BM3369":4295290368,
	   "BM3370":3101960704}

markerRadius = 6
markerRadius_bbox = 12

def _process_input(f):
    f_name = f.split('.')[0]
    
    inpath = "{}/{}".format(datadir, f)
    outpath = "{}/{}".format(blobdir, 'processed_' + f_name + '.h5')

    outfile = h5py.File(outpath)
    outshape = None

    try:
        outshape = outfile["/data"].shape
        print("  Skipping data blob generation... exists")
    except:
        print("  Generating data blob...")
        infile = h5py.File(inpath, 'r')

        print("    Loading '{}:/data'...".format(f))
        data = np.empty(infile["/data"].shape, dtype=np.float32)
        data[:] = infile["/data"]

        if np.sum(data) < 1.0:
	    print("    Empty file, removing " + outpath)
	    outfile.close()
	    os.remove(outpath)
	    return 0

        print("    Normalizing...")
	max_val = np.max(data)
	if f[:6] in max_int.keys():
	    max_val = math.log(max_int[f[:6]])
	
        data = (data - np.min(data)) / (max_val - np.min(data))
        outshape = data.shape
        blob = np.reshape(data, (1, 1, data.shape[0], data.shape[1]))

        print "    Saving '{}:/data'...".format(outpath)
        dset = outfile.create_dataset("/data",
                                      blob.shape,
                                      chunks=True,
                                      compression='gzip',
                                      dtype=np.float32,
                                      data=blob)
        dset.attrs["rt_min"] = infile["/data"].attrs["rt_min"]
        dset.attrs["rt_max"] = infile["/data"].attrs["rt_max"]
        dset.attrs["rt_step_size"] = infile["/data"].attrs["rt_step_size"]
        dset.attrs["mz_min"] = infile["/data"].attrs["mz_min"]
        dset.attrs["mz_max"] = infile["/data"].attrs["mz_max"]
        dset.attrs["mz_step_size"] = infile["/data"].attrs["mz_step_size"]


    try:
        outfile["/label"]
        outfile["/label_weight"]
        print("  Skipping label and label_weight blob generation... exist")
    except:
        print("  Generating label and label_weight blobs...")
        infile = h5py.File(inpath, 'r')

        # creating markerStrel
        markerStrel = np.zeros((2 * markerRadius + 1, 2 * markerRadius + 1))
        gridX = np.outer(np.arange(-markerRadius, markerRadius + 1, 1),
                        np.ones((1, markerStrel.shape[1])))

        gridY = np.outer(np.ones((1, markerStrel.shape[0])),
                        np.arange(-markerRadius, markerRadius + 1, 1))

        markerStrel[gridX * gridX + gridY * gridY <= markerRadius * markerRadius] = 1

        # labels with a radius of markerRadius
        print("    Loading '{}:/label'...".format(inpath))
        data = np.empty(infile["/label"].shape, dtype=np.float32)
        data[:] = infile["/label"]
        featurePositions = np.nonzero(data)
        data = np.zeros(outshape, dtype=np.float32)
        data[featurePositions] = 1

        labels = ndimage.morphology.binary_dilation(data, structure=markerStrel)

        blob = np.reshape(labels, (1, 1, labels.shape[0], labels.shape[1]))

        print("    Saving '{}:/label'...".format(outpath))
        outfile.create_dataset("/label",
                               blob.shape,
                               chunks=True,
                               compression="gzip",
                               dtype=np.float32,
                               data=blob)

        # label weights with a radius of markerRadius around the labels
        weights = 1 - ndimage.morphology.binary_dilation(labels, structure=markerStrel) + labels

        blob = np.reshape(weights, (1, 1, weights.shape[0], weights.shape[1]))

        print("    Saving '{}:/label_weight'...".format(outpath))
        outfile.create_dataset("/label_weight",
                               blob.shape,
                               chunks=True,
                               compression="gzip",
                               dtype=np.float32,
                               data=blob)

    """try:
        outfile["/bbox"]
        outfile["/bbox_weight"]
        print("  Skipping bbox and bbox_weight blob generation... exist")
    except:
        print("  Generating bbox and bbox_weight blobs...")
        infile = h5py.File(inpath, 'r')

        # bounding box as vectors
        print("    Loading '{}:/feature'...".format(inpath))
        data = np.empty(infile["/feature"].shape, dtype=np.float32)
        data[:] = infile["/feature"]

        blob_shape = (1, data.shape[0], data.shape[1], data.shape[2])
        blob = np.empty(blob_shape, dtype=np.float32)

        blob = np.reshape(data, blob_shape)

        print("    Saving '{}:/bbox'...".format(outpath))
        outfile.create_dataset("/bbox",
                               blob.shape,
                               chunks=True,
                               compression="gzip",
                               dtype=np.float32,
                               data=blob)


        # bounding box or feature weights
        print("    Loading '{}:/label'...".format(inpath))

        # creating markerStrel
        markerStrel = np.zeros((2 * markerRadius_bbox + 1, 2 * markerRadius_bbox + 1))
        gridX = np.outer(np.arange(-markerRadius_bbox, markerRadius_bbox + 1, 1),
                         np.ones((1, markerStrel.shape[1])))

        gridY = np.outer(np.ones((1, markerStrel.shape[0])),
                         np.arange(-markerRadius_bbox, markerRadius_bbox + 1, 1))

        markerStrel[gridX * gridX + gridY * gridY < markerRadius_bbox * markerRadius_bbox] = 1

        data = np.empty(infile["/label"].shape, dtype=np.float32)
        data[:] = infile["/label"]
        featurePositions = np.nonzero(data)
        data = np.zeros(outshape, dtype=np.float32)
        data[featurePositions] = 1

        weights = ndimage.morphology.binary_dilation(data, structure=markerStrel)

        # Bounding Box has 4 channels, weights also have 4 channels
        blob_shape = (1, 4, weights.shape[0], weights.shape[1])
        blob = np.empty(blob_shape, dtype=np.float32)

        for i in range(4):
            blob[0, i, :, :] = weights

        print("    Saving'{}:/bbox_weight'...".format(outpath))
        outfile.create_dataset("/bbox_weight",
                               blob.shape,
                               chunks=True,
                               compression="gzip",
                               dtype=np.float32,
                               data=blob)"""

    try:
        outfile["/label_ch_1"]
        print("  Skipping label_ch_1 blob generation... exists")
    except:
        print("  Generating label_ch_1 blob...")
        infile = h5py.File(inpath, 'r')

        print("    Loading '{}:/label_ch_1'...".format(f))
        data = np.empty(infile["/label_ch_1"].shape, dtype=np.float32)
        data[:] = infile["/label_ch_1"]

        blob = np.reshape(data, (1, 1, data.shape[0], data.shape[1]))

        print "    Saving '{}:/label_ch_1'...".format(outpath)
        dset = outfile.create_dataset("/label_ch_1",
                                      blob.shape,
                                      chunks=True,
                                      compression='gzip',
                                      dtype=np.float32,
                                      data=blob)


    """try:
        outfile["/label_ch_all"]
        print("  Skipping label_ch_all blob generation... exists")
    except:
        print("  Generating label_ch_all blob...")
        infile = h5py.File(inpath, 'r')

        print("    Loading '{}:/label_ch_all'...".format(f))
        data = np.empty(infile["/label_ch_all"].shape, dtype=np.float32)
        data[:] = infile["/label_ch_all"]

        blob = np.reshape(data, (1, 1, data.shape[0], data.shape[1]))

        print "    Saving '{}:/label_ch_all'...".format(outpath)
        dset = outfile.create_dataset("/label_ch_all",
                                      blob.shape,
                                      chunks=True,
                                      compression='gzip',
                                      dtype=np.float32,
                                      data=blob)"""

    """try:
        outfile["/label_diff_ch"]
        print("  Skipping label_diff_ch blob generation... exists")
    except:
        print("  Generating label_diff_ch blob...")
        infile = h5py.File(inpath, 'r')

        print("    Loading '{}:/label_diff_ch'...".format(f))
        data = np.empty(infile["/label_diff_ch"].shape, dtype=np.float32)
        data[:] = infile["/label_diff_ch"]

        blob = np.reshape(data, (1, 1, data.shape[0], data.shape[1])) #, data.shape[2]))

        print "    Saving '{}:/label_diff_ch'...".format(outpath)
        dset = outfile.create_dataset("/label_diff_ch",
                                      blob.shape,
                                      chunks=True,
                                      compression='gzip',
                                      dtype=np.float32,
                                      data=blob)"""
    return 0


if __name__ == '__main__':
    files = [f for f in os.listdir("{}".format(datadir)) if f.endswith(".h5")]

    # create files in parallel
    pool = Pool(processes=n_processes)
    pool.map(_process_input, files)

