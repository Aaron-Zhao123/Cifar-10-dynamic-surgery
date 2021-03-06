import os
import train
import sys
from shutil import copyfile

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(p['cov1'] * 10))
    name += 'cov' + str(int(p['cov2'] * 10))
    name += 'fc' + str(int(round(p['fc1'] * 10)))
    name += 'fc' + str(int(p['fc2'] * 10))
    name += 'fc' + str(int(p['fc3'] * 10))
    return name

acc_list = []
count = 0
retrain = 0
parent_dir = 'assets/'
# lr = 1e-5
lr = 1e-4
crates = {
    'cov1': 0.,
    'cov2': 1.6,
    'fc1': 2.1,
    'fc2': 1.6,
    'fc3': 0.
}
iter_cnt = 1
retrain_cnt = 0
roundrobin = 0
parent_dir = './assets/'
# Prune
while (crates['fc3'] < 1.5):
    count = 0
    model_tag = 0
    iter_cnt = 0
    while(iter_cnt <= 7):
        param = [
            ('-first_time', False),
            ('-train', False),
            ('-prune', True),
            ('-lr', lr),
            ('-parent_dir', parent_dir),
            ('-iter_cnt',iter_cnt),
            ('-cRates',crates)
            ]
        _ = train.main(param)

        # pruning saves the new models, masks
        while (retrain < 4):
            if (retrain == 2):
                lr = 5e-5
            elif (retrain == 3):
                lr = 1e-5
            else:
                lr = 1e-4
            # TRAIN
            param = [
                ('-first_time', False),
                ('-train', True),
                ('-prune', False),
                ('-lr', lr),
                ('-parent_dir', parent_dir),
                ('-iter_cnt',iter_cnt),
                ('-cRates',crates)
                ]
            _ = train.main(param)

            # TEST
            param = [
                ('-first_time', False),
                ('-train', False),
                ('-prune', False),
                ('-lr', lr),
                ('-parent_dir', parent_dir),
                ('-iter_cnt',iter_cnt),
                ('-cRates',crates)
                ]
            acc = train.main(param)

            if (acc > 0.823):
                lr = 1e-4
                retrain = 0
                break
            else:
                retrain = retrain + 1
        if (acc > 0.823 or iter_cnt == 7):
            file_name = compute_file_name(crates)
            crates['cov1'] = crates['cov1'] + 0.2
            crates['fc3'] = crates['fc3'] + 0.2
            # crates['fc1'] = crates['fc1'] + 0.2
            acc_list.append((crates,acc))
            param = [
                ('-first_time', False),
                ('-train', False),
                ('-prune', False),
                ('-lr', lr),
                ('-parent_dir', parent_dir),
                ('-iter_cnt',iter_cnt),
                ('-cRates',crates),
                ('-save', True),
                ('-org_file_name', file_name)
                ]
            _ = train.main(param)
            break
        else:
            iter_cnt = iter_cnt + 1

    if (iter_cnt > 7):
        iter_cnt = iter_cnt - 1
    print('accuracy summary: {}'.format(acc_list))

print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
