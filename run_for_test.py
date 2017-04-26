import os
import train
import sys
from shutil import copyfile

def compute_file_name(pcov, pfc):
    name = ''
    name += 'cov' + str(int(pcov[0] * 10))
    name += 'cov' + str(int(pcov[1] * 10))
    name += 'fc' + str(int(round(pfc[0] * 10)))
    name += 'fc' + str(int(pfc[1] * 10))
    name += 'fc' + str(int(pfc[2] * 10))
    return name

acc_list = []
count = 0
retrain = 0
parent_dir = 'assets/'
# lr = 1e-5
lr = 1e-4
crates = {
    'cov1': 0.,
    'cov2': 0.,
    'fc1': 1.6,
    'fc2': 0.,
    'fc3': 0.
}
iter_cnt = 1
retrain_cnt = 0
roundrobin = 0
with_biases = False
prev_parent_dir = './assets/' + 'cr' + 'fc1v' + str(int(crates['fc1']*100)) + '/'
parent_dir = './assets/' + 'cr' + 'fc1v' + str(int(crates['fc1']*100)) + '/'
iter_cnt = 0
# TEST
param = [
    ('-first_time', False),
    ('-train', False),
    ('-prune', False),
    ('-lr', lr),
    ('-with_biases', with_biases),
    ('-parent_dir', parent_dir),
    ('-iter_cnt',iter_cnt),
    ('-cRates',crates)
    ]
acc = train.main(param)
print('accuracy is {}'.format(acc))
