import h5py
import sys
sys.path.insert(0, '/home/urrehmaz/github/caffe-unet/python')
import numpy as np
import caffe
import os
from os.path import isfile, join

path = "PeptideLabelDetection"    # snapshot directory
input_file = "../caffeBlobs/data/debug_file.h5"    # training file
niters = 100000    # max iterations



# read file
f = h5py.File(input_file,"r")

data = f['/data'][:]
label = f['/label'][:]
label_weight= f['/label_weight'][:]

f.close()

# look for snapshots
snapshots = [snp for snp in os.listdir(path) if isfile(join(path, snp))]
solver_snapshots = [slv for slv in snapshots if slv.endswith("solverstate.h5")]

snap_iter = len(solver_snapshots) * 100    # using info that snapshots saved after every 100 iters

# set GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('PeptideLabelDetection-solver.prototxt')

solver.net.blobs['data'].data[...] = data[:]
solver.net.blobs['label'].data[...] = label[:]
solver.net.blobs['label_weight'].data[...] = label[:]

# if snapshot are present, load snapshot
if snap_iter > 0:
    last_snap = "PeptideLabelDetection/PeptideLabelDetection_snapshot_iter_" + str(snap_iter) + ".solverstate.h5"
    print "Continuing from snapshot " + last_snap
    solver.restore(last_snap)

for i in xrange(niters):
    solver.step(1)
    
    loss_i = solver.net.blobs['loss'].data
