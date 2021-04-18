import csv
import os
import pickle

data_dir = 'SeedCup2019_pre'
train_csv = os.path.join(data_dir, 'SeedCup_pre_train.csv')

with open(train_csv, 'r') as csvfile:
    lines = csv.reader(csvfile)
    train_data_row = []

    for i in lines:
        train_data_row.append(i)

for data in train_data_row:
    data = ''.join(data)

train_data = []
title = []

for i in range(len(train_data_row)):
    if(i!=0):
        train_data.append(train_data_row[i][0].split('\t'))
    else:
        title.append(train_data_row[i][0].split('\t'))

all_ck = {}
for data in train_data:
    if (int(data[13])) not in all_ck:
        all_ck[int(data[13])] = 1
    else:
        all_ck[int(data[13])] += 1

all_ck_l = []

for i in all_ck:
    all_ck_l.append(all_ck[i])

ck_i = all_ck_l.index(max(all_ck_l))

buffer = 0

for i in all_ck:
    if buffer == ck_i:
        print('最常用仓库', i)
    buffer += 1

buffer = 0


all_wl = {}
for data in train_data:
    if (int(data[12])) not in all_wl:
        all_wl[int(data[12])] = 1
    else:
        all_wl[int(data[12])] += 1

all_wl_l = []

for i in all_wl:
    all_wl_l.append(all_wl[i])

wl_i = all_wl_l.index(max(all_wl_l))
for i in all_wl:
    if buffer == wl_i:
        print('最多物流公司', i)
    buffer += 1

buffer = 0


all_fh = {}
for data in train_data:
    if (int(data[14])) not in all_fh:
        all_fh[int(data[14])] = 1
    else:
        all_fh[int(data[14])] += 1

all_fh_l = []

for i in all_fh:
    all_fh_l.append(all_fh[i])
fh_i = all_fh_l.index(max(all_fh_l))
for i in all_fh:
    if buffer == fh_i:
        print('最多发货地址', i)
    buffer += 1




