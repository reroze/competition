import csv
import os
import pickle

#for data in train_data:
    #print(data[11])#商家id
#data[10]是商家id，data【12】是物流公司id，data【13】是仓库id,data[6]是商品第一目类,data[14]发货省份
from typing import Dict, List


class tongji():
    def __init__(self):
        self.data_12 = {}
        self.data_12_num = 0
        self.data_13 = {}
        self.data_13_num = 0
        self.data_14 = {}
        self.data_14_num = 0

    def record(self, data):
        if(int(data[12]) not in self.data_12):
            self.data_12[int(data[12])] = 1
        else:
            self.data_12[int(data[12])] += 1
        if (int(data[13]) not in self.data_13):
            self.data_13[int(data[13])] = 1
        else:
            self.data_13[int(data[13])] += 1
        if (int(data[14]) not in self.data_14):
            self.data_14[int(data[14])] = 1
        else:
            self.data_14[int(data[14])] += 1
        self.data_12_num += 1
        self.data_13_num += 1
        self.data_14_num += 1

    def zhengli(self):
        for i in self.data_12:
            self.data_12[i] = self.data_12[i] / self.data_12_num
        for i in self.data_13:
            self.data_13[i] = self.data_13[i] / self.data_13_num
        for i in self.data_14:
            self.data_14[i] = self.data_14[i] / self.data_14_num
    def printf(self):
        print('物流公司概率', self.data_12)
        print('仓库概率', self.data_13)
        print('发货省份' ,self.data_14)
store = tongji()

data_dir = 'SeedCup2019_pre'
train_csv = os.path.join(data_dir, 'SeedCup_pre_train.csv')

test_csv = os.path.join(data_dir, 'SeedCup_pre_test.csv')

with open(test_csv, 'r') as csvfile:
    lines = csv.reader(csvfile)
    test_data_row = []

    for i in lines:
        test_data_row.append(i)

for data in test_data_row:
    data = ''.join(data)

test_data = []

for i in range(len(test_data_row)):
    if(i!=0):
        test_data.append(test_data_row[i][0].split('\t'))
    #else:
        #title.append(train_data_row[i][0].split('\t'))
sum9 = 0
test_id = []
for i in test_data:
    if int(i[10]) ==58 :
        sum9 += 1

print('sum9', sum9)


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


print('title [10]:', title[0][10])
#print('title [11]:', title[0][11])
print('title [12]:', title[0][12])
print('title [13]', title[0][13])
print('title [14]', title[0][14])

ys_sel2lg = {}

#for data in train_data:
    #print(data[11])#商家id
#data[10]是商家id，data【12】是物流公司id，data【13】是仓库id,data[6]是商品第一目类,data[14]发货省份,data[16]是收货省份

#sum=0
'''
for data in train_data:
    #if int(data[13]) == 1:
        #print(data[13])
        #sum += 1
    store.record(data)
    if int(data[10]) not in ys_sel2lg:
        ys_sel2lg[int(data[10])] = {int(data[6]) : {int(data[12]) : {int(data[16]) : [int(data[13])]}}}

    else:
        if int(data[6]) not in ys_sel2lg[int(data[10])]:
            ys_sel2lg[int(data[10])][int(data[6])]={int(data[12]) : {int(data[16]) : [int(data[13])]}}
            #if data[10] == '766':
                #print('766, 字典， 物流公司' ,ys_sel2lg[data[10]], data[12])
        elif int(data[12]) not in ys_sel2lg[int(data[10])][int(data[6])]:
            ys_sel2lg[int(data[10])][int(data[6])][int(data[12])]={int(data[16]) : [int(data[13])]}
        elif int(data[16]) not in ys_sel2lg[int(data[10])][int(data[6])][int(data[12])]:
            ys_sel2lg[int(data[10])][int(data[6])][int(data[12])][int(data[16])] = [int(data[13])]
        elif int(data[13]) not in ys_sel2lg[int(data[10])][int(data[6])][int(data[12])][int(data[16])]:
            ys_sel2lg[int(data[10])][int(data[6])][int(data[12])][int(data[16])].append(int(data[13]))
            #if(len(ys_sel2lg[int(data[10])][int(data[6])][int(data[12])][int(data[14])]) > 1):
                #print('haha', ys_sel2lg[int(data[10])][int(data[6])][int(data[12])][int(data[14])])

'''
ys_seq2wl = {}

sum1=0

for data in train_data:
    if int(data[10]) not in ys_seq2wl:
        ys_seq2wl[int(data[10])] = {int(data[6]) : [[int(data[12])],[1]]}
        #sum+=1
    elif int(data[6]) not in ys_seq2wl[int(data[10])]:
        ys_seq2wl[int(data[10])][int(data[6])] = [[int(data[12])], [1]]
        #sum+=1
    elif int(data[12]) not in ys_seq2wl[int(data[10])][int(data[6])][0]:
        ys_seq2wl[int(data[10])][int(data[6])][0].append(int(data[12]))
        ys_seq2wl[int(data[10])][int(data[6])][1].append(1)
        #sum+=1
    elif int(data[12]) in ys_seq2wl[int(data[10])][int(data[6])][0]:
        for i in range(len(ys_seq2wl[int(data[10])][int(data[6])][0])):
            #print(len(ys_seq2wl[int(data[10])][int(data[6])][0]))
            if(int(data[12]) == ys_seq2wl[int(data[10])][int(data[6])][0][i]):
                #print('haha')
                ys_seq2wl[int(data[10])][int(data[6])][1][i]+=1
                #print(ys_seq2wl[int(data[10])][int(data[6])])
            #print('haha')
        #sum+=1

ys_seq2fh = {}#商家-商品-发货省份

for data in train_data:#发货省份就一对一
    if int(data[10]) not in ys_seq2fh:
        ys_seq2fh[int(data[10])] = {int(data[6]) : [[int(data[14])],[1]]}
        #sum+=1
    elif int(data[6]) not in ys_seq2fh[int(data[10])]:
        ys_seq2fh[int(data[10])][int(data[6])] = [[int(data[14])], [1]]
        #sum+=1
    elif int(data[14]) not in ys_seq2fh[int(data[10])][int(data[6])][0]:
        ys_seq2fh[int(data[10])][int(data[6])][0].append(int(data[14]))
        ys_seq2fh[int(data[10])][int(data[6])][1].append(1)
        #sum+=1
    elif int(data[14]) in ys_seq2fh[int(data[10])][int(data[6])][0]:
        for i in range(len(ys_seq2fh[int(data[10])][int(data[6])][0])):
            #print(len(ys_seq2wl[int(data[10])][int(data[6])][0]))
            if(int(data[14]) == ys_seq2fh[int(data[10])][int(data[6])][0][i]):
                #print('haha')
                ys_seq2fh[int(data[10])][int(data[6])][1][i]+=1
                #print(ys_seq2wl[int(data[10])][int(data[6])])
            #print('haha')
        #sum+=1
sum2=0
#print(ys_seq2wl)
#print(ys_seq2fh)
for data in ys_seq2fh:
    for i in ys_seq2fh[data]:
        if(len(ys_seq2fh[data][i][0])>1):
            sum2+=1
            print(ys_seq2fh[data][i])
ys_sh2ck = {}


ys_seq2wl_gv = {}
#print(sum)
print('sum2',sum2)

for data in train_data:
    if int(data[10]) not in ys_sh2ck:
        ys_sh2ck[int(data[10])] = {int(data[16]) : [[int(data[13])], [1]]}
    elif int(data[16]) not in ys_sh2ck[int(data[10])]:
        ys_sh2ck[int(data[10])][int(data[16])] = [[int(data[13])], [1]]
    elif int(data[13]) not in ys_sh2ck[int(data[10])][int(data[16])][0]:
        ys_sh2ck[int(data[10])][int(data[16])][0].append(int(data[13]))
        ys_sh2ck[int(data[10])][int(data[16])][1].append(1)
    elif int(data[13]) in ys_sh2ck[int(data[10])][int(data[16])][0]:
        for i in range(len(ys_sh2ck[int(data[10])][int(data[16])][0])):
            if(int(data[13]) == ys_sh2ck[int(data[10])][int(data[16])][0][i]):
                #print('haha')
                ys_sh2ck[int(data[10])][int(data[16])][1][i] += 1
            #print('haha')


print(ys_sh2ck)#商家-收货-对应仓库
#print(ys_sh2ck[929][30][1])
#buffer = sum(ys_sh2ck[929][30][1])
buffer = 0
print('ys_seq2fh', ys_seq2fh)

for i in ys_sh2ck:
    for j in ys_sh2ck[i]:
        buffer = sum(ys_sh2ck[i][j][1])
        for k in range(len(ys_sh2ck[i][j][1])):
            ys_sh2ck[i][j][1][k] =  ys_sh2ck[i][j][1][k]/buffer
        buffer = 0

for i in ys_seq2fh:
    for j in ys_seq2fh[i]:
        buffer = sum(ys_seq2fh[i][j][1])
        #print('buffer', buffer)
        for k1 in range(len(ys_seq2fh[i][j][1])):
            #print('k1', k1)
            #print('i', i)
            #print('j', j)
            ys_seq2fh[i][j][1][k1] = ys_seq2fh[i][j][1][k1]/buffer
        buffer = 0


for i in ys_seq2wl:
    for j in ys_seq2wl[i]:
        buffer = sum(ys_seq2wl[i][j][1])
        #print('buffer', buffer)
        for k1 in range(len(ys_seq2wl[i][j][1])):
            #print('k1', k1)
            #print('i', i)
            #print('j', j)
            ys_seq2wl[i][j][1][k1] = ys_seq2wl[i][j][1][k1]/buffer
        buffer = 0


#print(ys_seq2fh[67][16][1])

print('概率化:商家-收货-对应仓库', ys_sh2ck)
print('概率化：商家-商品-对应发货省份', ys_seq2fh)
print('概率化：商家-商品目类-对应物流公司', ys_seq2wl)
sum3=0

#for data in ys_sh2ck:
    #for i in ys_sh2ck[data]:
        #if(len(ys_sh2ck[data][i][0])>1):
            #sum3+=1
            #print(ys_sh2ck[data][i])

#print(ys_sh2ck)

#print('sum3', sum3)
#print(ys_sel2lg)
#store.zhengli()
#store.printf()
#print(ys_sel2lg['766'])
#print(sum)

#仓库--路线
#路线-发货地址，收货地址
#发货地址 商家-商品目类，收货地址--已知
#【商家，商品目类，收货省份-仓库】 {商家：{商品目类：{收货省份：【【仓库】【次数】】}}}
#【商家，收货省份-仓库】 {商家：{收货省份：【【仓库】【次数】】}}
#仓库--收货地址，商家，

save_dir_name = 'data'
'''
print('概率化:商家-收货-对应仓库', ys_sh2ck)
print('概率化：商家-商品-对应发货省份', ys_seq2fh)
print('概率化：商家-商品目类-对应物流公司', ys_seq2wl)
'''

save_ys_sh2ck = os.path.join(save_dir_name, '_ys_sh2ck.pkl')
save_ys_seq2fh = os.path.join(save_dir_name, '_ys_seq2fh.pkl')
save_ys_seq2wl = os.path.join(save_dir_name, '_ys_seq2wl.pkl')

with open(save_ys_sh2ck, 'wb') as f:
    pickle.dump(ys_sh2ck, f)

with open(save_ys_seq2fh, 'wb') as f:
    pickle.dump(ys_seq2fh, f)

with open(save_ys_seq2wl, 'wb') as f:
    pickle.dump(ys_seq2wl, f)

all_ck = {}
for data in train_data:
    if(int(data[13])) not in all_ck:
        all_ck[int(data[13])] = 1
    else:
        all_ck[int(data[13])] += 1

all_ck_l = []

for i in all_ck:
    all_ck_l.append(all_ck[i])
print(all_ck[all_ck_l.index(max(all_ck_l))])


all_wl = {}
for data in train_data:
    if(int(data[12])) not in all_wl:
        all_wl[int(data[12])] = 1
    else:
        all_wl[int(data[12])] += 1

all_wl_l = []

for i in all_wl:
    all_wl_l.append(all_wl[i])
print(all_wl[all_wl_l.index(max(all_wl_l))])


all_fh = {}
for data in train_data:
    if(int(data[14])) not in all_fh:
        all_fh[int(data[14])] = 1
    else:
        all_fh[int(data[14])] += 1

all_fh_l = []

for i in all_fh:
    all_fh_l.append(all_fh[i])
print(all_fh[all_fh_l.index(max(all_fh_l))])




