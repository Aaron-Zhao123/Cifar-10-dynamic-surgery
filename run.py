import os
import train
import sys
from shutil import copyfile

def compute_file_name(pcov, pfc):
    name = ''
    name += 'cov' + str(int(pcov[0] * 10))
    name += 'cov' + str(int(pcov[1] * 10))
    name += 'fc' + str(int(pfc[0] * 10))
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
    'fc1': 0.,
    'fc2': 0.,
    'fc3': 0.
}
retrain_cnt = 0
roundrobin = 0
with_biases = False
prev_parent_dir = './assets/' + 'cr' + 'fc1v' + str(int(crates['fc1']*100)) + '/'
crates['fc1'] = crates['fc1'] + 1.
# Prune
while (crates['fc1'] < 3.5):
    parent_dir = './assets/' + 'cr' + 'fc1v' + str(int(crates['fc1']*100)) + '/'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        src_dir = prev_parent_dir+'weight_crate'+str(count)+'.pkl'
        dest_dir = parent_dir + 'weight_crate0.pkl'
        copyfile(src_dir,dest_dir)
        src_dir = prev_parent_dir+'mask_crate'+str(count)+'.pkl'
        dest_dir = parent_dir + 'mask_crate0.pkl'
        copyfile(src_dir,dest_dir)
    count = 0
    model_tag = 0
    iter_cnt = 0
    while(iter_cnt <= 7):
        while (retrain < 10):
            param = [
                ('-first_time', False),
                ('-train', False),
                ('-prune', True),
                ('-lr', lr),
                ('-with_biases', with_biases),
                ('-parent_dir', parent_dir),
                ('-iter_cnt',iter_cnt),
                ('-cRates',crates)
                ]
            _ = train.main(param)

            # pruning saves the new models, masks

            # TRAIN
            param = [
                ('-first_time', False),
                ('-train', True),
                ('-prune', False),
                ('-lr', lr),
                ('-with_biases', with_biases),
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
                ('-with_biases', with_biases),
                ('-parent_dir', parent_dir),
                ('-iter_cnt',iter_cnt),
                ('-cRates',crates)
                ]
            acc = train.main(param)
            hist.append((pcov, pfc, acc))
            # pcov[1] = pcov[1] + 10.
            if (acc > 0.823):
                lr = 1e-4
                retrain = 0
                acc_list.append((pcov,pfc,acc))
                break
            else:
                retrain = retrain + 1
        iter_cnt = iter_cnt + 1
    print('accuracy summary: {}'.format(acc_list))
    print (acc)

print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
with open("acc_cifar.txt", "w") as f:
    for item in acc_list:
        f.write("{} {} {}\n".format(item[0],item[1],item[2]))
