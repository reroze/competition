import pickle
import os
import csv
import random
save_dir = 'SeedCup2019_pre'
test_file = 'SeedCup_pre_test.csv'
pkl_save_dir = 'data'

f = open(os.path.join(pkl_save_dir, '_ys_sh2ck.pkl'), 'rb')
ys_sh2ck = pickle.load(f)
f.close()

f = open(os.path.join(pkl_save_dir, '_ys_seq2fh.pkl'), 'rb')
ys_seq2fh = pickle.load(f)
f.close()

f = open(os.path.join(pkl_save_dir, '_ys_seq2wl.pkl'), 'rb')
ys_seq2wl = pickle.load(f)
f.close()

data_dir = 'SeedCup2019_pre'

test_csv = os.path.join(data_dir, 'SeedCup_pre_test.csv')

with open(test_csv, 'r') as csvfile:
    lines = csv.reader(csvfile)
    test_data_row = []

    for i in lines:
        test_data_row.append(i)

for data in test_data_row:
    data = ''.join(data)

test_data = []
title = []

for i in range(len(test_data_row)):
    if(i!=0):
        test_data.append(test_data_row[i][0].split('\t'))
    else:
        title.append(test_data_row[i][0].split('\t'))

'''
ys_sh2ck #商家id，收货地址-决定仓库
ys_seq2fh #商家id，商品目类-决定发货省份
ys_seq2wl #商家id， 商品目类-决定物流公司
'''
#print(title)['uid', 'plat_form', 'biz_type', 'create_time', 'payed_time', 'product_id', 'cate1_id', 'cate2_id', 'cate3_id', 'preselling_shipped_time', 'seller_uid', 'company_name', 'rvcr_prov_name', 'rvcr_city_name']
'''
先来预测仓库
'''

def get_ck(ys_sh2ck, i):
    wuliu = []
    for j in ys_sh2ck[i]:
        for k in range(len(ys_sh2ck[i][j][0])):
            if ys_sh2ck[i][j][0][k] not in wuliu:
                wuliu.append(ys_sh2ck[i][j][0][k])
    value = random.choice(wuliu)
    return value



prb_ys_sh2ck = {}
all_prb = {}

for i in test_data:
    if int(i[10]) not in prb_ys_sh2ck:
        prb_ys_sh2ck[int(i[10])] = {int(i[12]) : 10}
    elif int(i[12]) not in prb_ys_sh2ck[int(i[10])] :
        prb_ys_sh2ck[int(i[10])][int(i[12])] = 10
'''
for i in ys_sh2ck:
    for j in ys_sh2ck[i]:
        if(i==929):
            print('ceshi', ys_sh2ck[i][j][1])
            '''

for i in prb_ys_sh2ck:
    if i in ys_sh2ck:
        for j in prb_ys_sh2ck[i]:
            if j in ys_sh2ck[i]:
                prb_ys_sh2ck[i][j] = ys_sh2ck[i][j][0][ys_sh2ck[i][j][1].index(max(ys_sh2ck[i][j][1]))]
            else:
                prb_ys_sh2ck[i][j] = get_ck(ys_sh2ck,i)

print(prb_ys_sh2ck)

'''
ys_seq2fh #商家id，商品目类-决定发货省份 #在随机预测时，可引入总的次数而不是概率来提高准确率
再来预测省份
'''
prb_ys_seq2fh = {}

for i in test_data:
    if int(i[10]) not in prb_ys_seq2fh:
        prb_ys_seq2fh[int(i[10])] = {int(i[6]) : 4}
    elif int(i[6]) not in prb_ys_seq2fh[int(i[10])] :
        prb_ys_seq2fh[int(i[10])][int(i[6])] = 4

for i in prb_ys_seq2fh:
    if i in ys_seq2fh:
        for j in prb_ys_seq2fh[i]:
            if j in ys_seq2fh[i]:
                prb_ys_seq2fh[i][j] = ys_seq2fh[i][j][0][ys_seq2fh[i][j][1].index(max(ys_seq2fh[i][j][1]))]
            else:
                prb_ys_seq2fh[i][j] = get_ck(ys_seq2fh, i)

print(prb_ys_seq2fh)


'''
ys_seq2wl #商家id， 商品目类-决定物流公司
再来确定物流公司

'''

prb_ys_seq2wl = {}

for i in test_data:
    if int(i[10]) not in prb_ys_seq2wl:
        prb_ys_seq2wl[int(i[10])] = {int(i[6]) : 9}
    elif int(i[6]) not in prb_ys_seq2wl[int(i[10])] :
        prb_ys_seq2wl[int(i[10])][int(i[6])] = 9

for i in prb_ys_seq2wl:
    if i in ys_seq2wl:
        for j in prb_ys_seq2wl[i]:
            if j in ys_seq2wl[i]:
                prb_ys_seq2wl[i][j] = ys_seq2wl[i][j][0][ys_seq2wl[i][j][1].index(max(ys_seq2wl[i][j][1]))]
            else:
                prb_ys_seq2wl[i][j] = get_ck(ys_seq2wl, i)


#print(prb_ys_seq2wl[58])

save_dir_name = 'data'


save_prb_ys_sh2ck = os.path.join(save_dir_name, '_prb_ys_sh2ck.pkl')
save_prb_ys_seq2fh = os.path.join(save_dir_name, '_prb_ys_seq2fh.pkl')
save_prb_ys_seq2wl = os.path.join(save_dir_name, '_prb_ys_seq2wl.pkl')
'''
prb_ys_sh2ck #商家id，收货地址-决定仓库
prb_ys_seq2fh #商家id，商品目类-决定发货省份
prb_ys_seq2wl #商家id， 商品目类-决定物流公司
所有预测最后都为1个结果
'''

with open(save_prb_ys_sh2ck, 'wb') as f:
    pickle.dump(prb_ys_sh2ck, f)

with open(save_prb_ys_seq2fh, 'wb') as f:
    pickle.dump(prb_ys_seq2fh, f)

with open(save_prb_ys_seq2wl, 'wb') as f:
    pickle.dump(prb_ys_seq2wl, f)


#print(ys_sh2ck)

