"""Plot training and validation losses by parsing log files."""

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from os.path import isfile, join
import stat
import subprocess
import pandas as pd


log_dir = 'results'
network = 'PeptideLabelDetection'
sigma = 2000
sigma_test = 5

log_files = [f for f in os.listdir(log_dir) if isfile(join(log_dir, f))]
log_files = [f for f in log_files if f.startswith(network + '-train.o')]

log_files.sort()
n_logs = len(log_files)

# bash script for parsing log files
str = """#!/bin/bash
# NOTE: this script will be deleted afterwards
# Usage parse_log.sh caffe.log
# It creates the following two text files, each containing a table:
#     caffe.log.test (columns: '#Iters Seconds TestAccuracy TestLoss')
#     caffe.log.train (columns: '#Iters Seconds TrainingLoss LearningRate')


# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh /path/to/your.log"
exit
fi
LOG=`basename $1`
sed -n '/Iteration .* Testing net/,/Iteration *. loss/p' $1 > aux.txt
sed -i '/Waiting for data/d' aux.txt
sed -i '/prefetch queue empty/d' aux.txt
sed -i '/Iteration .* loss/d' aux.txt
sed -i '/Iteration .* lr/d' aux.txt
sed -i '/Train net/d' aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\\1/g' > aux0.txt    # use single back slash before 1 in a bash script
grep 'Test net output #0' aux.txt | awk '{print $11}' > aux1.txt
grep 'Test net output #1' aux.txt | awk '{print $11}' > aux2.txt


# Generating
echo '#Iters ch_1_loss label_loss'> $LOG.test
paste aux0.txt aux1.txt aux2.txt | column -t >> $LOG.test
rm aux.txt aux0.txt aux1.txt aux2.txt

# For extraction of time since this line contains the start time
grep '] Solving ' $1 > aux.txt
grep ', loss = ' $1 >> aux.txt
grep 'Train net output #' $1 >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\\1/g' > aux0.txt    # use single back slash before 1 in a bash script
grep ', loss = ' aux.txt | awk '{print $9} ' > aux1.txt
grep 'Train net output #0' aux.txt | awk '{print $11}' > aux2.txt
grep 'Train net output #1' aux.txt | awk '{print $11}' > aux3.txt

# Generating
echo '#Iters OverallLoss Train_ch1_loss Train_label_loss'> $LOG.train
paste aux0.txt aux1.txt aux2.txt aux3.txt | column -t >> $LOG.train
rm aux.txt aux0.txt aux1.txt aux2.txt  aux3.txt

"""
with open("parse_log.sh","w") as f:
    f.write(str)
os.chmod("parse_log.sh", stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)    # make the script executeable

### Generating loss files ###
for i in range(n_logs):
    # Using script caffe_unet/tools/extra/parse_log.sh and caffe_unet/tools/extra/extract_seconds.py
    cmd = './parse_log.sh ' + log_dir + '/' + log_files[i]
    print 'Executing: ' + cmd
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

#### Parsing Loss files ###
f_name = log_files[0] + '.train'
train_losses = np.loadtxt(f_name, dtype=np.float32, comments='#')
for i in range(1, n_logs):
    f_name = log_files[i] + '.train'
    tmp = np.loadtxt(f_name, dtype=np.float32, comments='#')
    train_losses = np.concatenate((train_losses, tmp), axis=0)


f_name = log_files[0] + '.test'
test_losses = np.loadtxt(f_name, dtype=np.float32, comments='#')
for i in range(1, n_logs):
    f_name = log_files[i] + '.test'
    tmp = np.loadtxt(f_name, dtype=np.float32, comments='#')
    if tmp.size <= 1:
        pass
    elif tmp.size == 3:
        test_losses = np.concatenate((test_losses, tmp.reshape(1,3)), axis=0)
    else:    
        test_losses = np.concatenate((test_losses, tmp), axis=0)

### Removing Loss files ###
for i in range(n_logs):
    cmd = 'rm -f ' + log_files[i] + '.train'
    print 'Executing: ' + cmd
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()


    cmd = 'rm -f ' + log_files[i] + '.test'
    print 'Executing: ' + cmd
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()


### Plotting Training Losses ###
legend = []
"""
# plotting overall training loss after convolution
filter = np.exp(-0.5 * (np.arange(-4*sigma, 4*sigma + 1, 1) / sigma)**2) / (math.sqrt(2 * math.pi) * sigma)
lossPadded = np.concatenate((np.ones(int(filter.shape[0] / 2)) * train_losses[0,1], train_losses[:,1], np.ones(int(filter.shape[0] / 2)) * train_losses[-1, 1]))
lossFiltered = np.convolve(lossPadded, filter, mode="valid")
plt.plot(train_losses[:,0], lossFiltered)
legend.append('Overall Loss')"""

# plotting training convex hull loss with convolution
filter = np.exp(-0.5 * (np.arange(-4*sigma, 4*sigma + 1, 1) / sigma)**2) / (math.sqrt(2 * math.pi) * sigma)
lossPadded = np.concatenate((np.ones(int(filter.shape[0] / 2)) * train_losses[0,2], train_losses[:,2], np.ones(int(filter.shape[0] / 2)) * train_losses[-1, 2]))
lossFiltered = np.convolve(lossPadded, filter, mode="valid")
plt.plot(train_losses[:,0], lossFiltered, color='g')
legend.append('Train CH Loss')


# plotting training label loss with convolution
#filter = np.exp(-0.5 * (np.arange(-4*sigma, 4*sigma + 1, 1) / sigma)**2) / (math.sqrt(2 * math.pi) * sigma)
lossPadded = np.concatenate((np.ones(int(filter.shape[0] / 2)) * train_losses[0,3], train_losses[:,3], np.ones(int(filter.shape[0] / 2)) * train_losses[-1, 3]))
lossFiltered = np.convolve(lossPadded, filter, mode="valid")
plt.plot(train_losses[:,0], lossFiltered, color='y')
legend.append('Train Label Loss')

### Plotting Test Losses ###
# test convex hull loss as scatter plot
sigma_test=1
filter = np.exp(-0.5 * (np.arange(-4*sigma_test, 4*sigma_test + 1, 1) / sigma_test)**2) / (math.sqrt(2 * math.pi) * sigma_test)
lossPadded = np.concatenate((np.ones(int(filter.shape[0] / 2)) * test_losses[0,1], test_losses[:,1], np.ones(int(filter.shape[0] / 2)) * test_losses[-1, 1]))
lossFiltered = np.convolve(lossPadded, filter, mode="valid")
plt.plot(test_losses[:,0], lossFiltered, color='b')
#plt.plot(test_losses[:, 0], test_losses[:, 1], marker='x', color='r')
legend.append('Valid CH Loss')

# test label loss as scatter plot
lossPadded = np.concatenate((np.ones(int(filter.shape[0] / 2)) * test_losses[0,2], test_losses[:,2], np.ones(int(filter.shape[0] / 2)) * test_losses[-1, 2]))
lossFiltered = np.convolve(lossPadded, filter, mode="valid")
plt.plot(test_losses[:,0], lossFiltered, color='r')
#plt.plot(test_losses[:, 0], test_losses[:, 2], marker='x', color='b')
legend.append('Valid Label Loss')

# delete generated bash script
os.remove("parse_log.sh")

plt.legend(legend)
plt.show()

