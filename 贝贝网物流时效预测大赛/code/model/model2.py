import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import math as ma
import random as rd
from config import Config
import numpy as np

'''
目标
rank 低于56准确率不管
60 0.96
61 0.97
62 0.98
'''

'''
目前可能的改进思路有？
loss的继续更改
1.低rank模型调试 看看能不能在0.90左右将rank降到50
2.高ontime 看看能不能调到 60.0
3.尝试计算数据的定位与处理
4.尝试新的网络结构？

'''

'''
design loss function
'''
class My_MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        #print('x_size', x.size())
        #x = list(x)
        #y = list(y)
        #x = x.reshape(x.size()[0], -1)
        #y = y.reshape(y.size()[0], -1)
        #print(x.size())
        '''
        if x.sum() > y.sum():
            #print(x.sum())
            a = torch.mean(torch.pow((x - y), 2))**0.5 * 2#mean是取中间值 #尽量让x小于y
            #print('a_size', a.size(), a)
            return a
        else:
            a = torch.mean(torch.pow((x - y), 2)) ** 0.5 # mean是取中间值 #尽量让x小于y
            #print('a_size', a.size(), a)
            return a
        '''

        times = 0
        drop = 0
        loss = 0
        loss = torch.tensor(loss)
        loss = loss.cuda()
        loss = torch.autograd.Variable(loss)
        loss = loss.float()
        #print('x_size', x.size())


        for i in range(x.size()[0]):
            if abs(x[i]-y[i]) <=1:
                if x[i] > y[i]:
                    loss += torch.pow((x[i] - y[i]), 2)*0.5
                    times += 1
                else:
                    loss += torch.pow((x[i] - y[i]), 2)*0.5
            else:
                if x[i] > y[i]:
                    loss += abs(x[i]-y[i])
                    times += 1
                else:
                    loss += abs(x[i]-y[i])
        loss = loss / x.size()[0]
        return loss

    '''
        for i in range(x.size()[0]):
            #if abs(x[i]-y[i])>5:
                #drop+=1
                #if drop>3:
                    #loss += torch.pow((x[i] - y[i]), 2)*ma.log(2+ma.exp(x[i]-y[i]))
                    #print('drop', drop)
            #else:
                if abs(x[i] - y[i]) >1:
                    if x[i]< y[i]:
                        loss += abs(x[i]-y[i]) * ma.log(2 + ma.exp((x[i] - y[i])))
                    else:
                        loss += abs(x[i] - y[i]) * ma.log(2 + ma.exp((x[i] - y[i])))
                else:
                    if x[i] < y[i]:
                        loss += torch.pow((x[i] - y[i]), 2)*ma.log(2+ma.exp((x[i]-y[i])))#提升ontime时，用rank最低的， 降低rank时，用ontime最高的 反复切换model进行训练， 先用低rank减少drop极端值
                        times += 1
                    else:
                        loss += torch.pow((x[i] - y[i]), 2) * ma.log(2 + ma.exp((x[i] - y[i])))



        loss = loss/x.size()[0]
        return loss
        '''


            
class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        # self.encoder_uid       = nn.Embedding(opt.uid_range, opt.EMBEDDING_DIM)#并没有使用uid因此一共才只有10个
        self.encoder_plat_form = nn.Embedding(opt.plat_form_range, opt.EMBEDDING_DIM)#做embedding
        self.encoder_biz_type = nn.Embedding(opt.biz_type_range, opt.EMBEDDING_DIM)
        self.encoder_product_id = nn.Embedding(opt.product_id_range, opt.EMBEDDING_DIM)

        self.encoder_cate1_id = nn.Embedding(opt.cate1_id_range, opt.EMBEDDING_DIM)
        self.encoder_cate2_id = nn.Embedding(opt.cate2_id_range, opt.EMBEDDING_DIM)
        self.encoder_cate3_id = nn.Embedding(opt.cate3_id_range, opt.EMBEDDING_DIM)

        self.encoder_seller_uid = nn.Embedding(opt.seller_uid_range, opt.EMBEDDING_DIM)
        self.encoder_company_name = nn.Embedding(opt.company_name_range, opt.EMBEDDING_DIM)
        self.encoder_rvcr_prov_name = nn.Embedding(opt.rvcr_prov_name_range, opt.EMBEDDING_DIM)
        self.encoder_rvcr_city_name = nn.Embedding(opt.rvcr_city_name_range, opt.EMBEDDING_DIM)
        self.encoder_lgst_company = nn.Embedding(opt.lgst_company_range, opt.EMBEDDING_DIM)
        self.encoder_warehouse_id = nn.Embedding(opt.warehouse_id_range, opt.EMBEDDING_DIM)
        self.encoder_shipped_prov_id = nn.Embedding(opt.shipped_prov_id_range, opt.EMBEDDING_DIM)

        self.FC_0_1 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(13 * opt.EMBEDDING_DIM, 1300),#输出的维度 1
            nn.BatchNorm1d(1300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.FC_0_2 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(900, 400),  # 输出的维度 1
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        #########如何 对小部分极端数据进行处理？

        self.Con1 = nn.Sequential(
            nn.Conv2d(1,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )

        self.Con2 = nn.Sequential(
            nn.Conv2d(8,1,3,1,1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

        self.Con = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(True),

            nn.Conv2d(in_channels=24, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8)


        )

        self.FC_R_0 = nn.Sequential(
            nn.Linear(1300, 2048),  # 输出的维度 1                                                     1300
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 输出的维度 1                                                       1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.FC_R_1 = nn.Sequential(
            nn.Linear(1600, 1000),  # 输出的维度 1                                                         768#
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 2                                                             512
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1600),  # 400到600 3                                                          768#res1
            nn.BatchNorm1d(768),
            nn.Dropout(0.5)
        )

        self.FC_R_2 = nn.Sequential(
            nn.Linear(768, 1024),  # 400到600 3                                                         768#
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 768),  # 400到600 4                                                          1024
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(768, 512),  # 400到600 5                                                           768#res2
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 768),  # 400到600 6                                                             512#
            nn.BatchNorm1d(768),
            nn.Dropout(0.5),
        )
        self.FC_R_3 = nn.Sequential(
            nn.Linear(768, 512),  # 400到600 7                                                            768
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 384),  # 400到600 8                                                           512#res3
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(384, 256),  # 400到600 9                                                            384
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 128),  # 400到600 10                                                           256
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 1)  # 600到1 11
        )

        self.FC_1_1 = nn.Sequential(

            #nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),

            nn.Linear(1300, 2048),  # 输出的维度 1                                                     1300
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 输出的维度 1                                                       1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 输出的维度 1                                                         768#
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),#400到600 2                                                             512
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1600),  # 400到600 3                                                          768#res1
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 2048),  # 400到600 3                                                         768#
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 400到600 4                                                          1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 5                                                           768#res2
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 6                                                             512#
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 7                                                            768
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 800),  # 400到600 8                                                           512#res3
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 600),  # 400到600 9                                                            384
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 400),  # 400到600 10                                                           256
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 1)#600到1 11                                                                   128
        )

        '''
            nn.Linear(1300, 1000),  # 输出的维度 1
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 800),  # 输出的维度 1
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 400),  # 输出的维度 1
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),  # 400到600 2
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 800),  # 400到600 3
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 1000),  # 400到600 3
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 800),  # 400到600 4
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 600),  # 400到600 5
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 800),  # 400到600 6
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 600),  # 400到600 7
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 400),  # 400到600 8
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 400),  # 400到600 9
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 100),  # 400到600 10
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(100, 1) 
        '''

        self.FC_1_2 = nn.Sequential(
            #  TODO change input dimension

            # nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),


            nn.Linear(1300, 2048),  # 输出的维度 1                                                     1300
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 输出的维度 1                                                       1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 输出的维度 1                                                         768#
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),#400到600 2                                                             512
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1600),  # 400到600 3                                                          768#res1
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 2048),  # 400到600 3                                                         768#
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 400到600 4                                                          1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 5                                                           768#res2
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 6                                                             512#
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 7                                                            768
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 800),  # 400到600 8                                                           512#res3
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 600),  # 400到600 9                                                            384
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 400),  # 400到600 10                                                           256
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 1)
            #600到1 11  #600到1 11  #600到1 11 #600到1 11  #600到1 #600到1 #600到1                                  6
        )

        self.FC_2_1 = nn.Sequential(

            # nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),

            nn.Linear(1300, 2048),  # 输出的维度 1                                                     1300
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 输出的维度 1                                                       1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 输出的维度 1                                                         768#
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 2                                                             512
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1600),  # 400到600 3                                                          768#res1
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 2048),  # 400到600 3                                                         768#
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 400到600 4                                                          1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 5                                                           768#res2
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 6                                                             512#
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 7                                                            768
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 800),  # 400到600 8                                                           512#res3
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 600),  # 400到600 9                                                            384
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 400),  # 400到600 10                                                           256
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 1)  # 600到1 11                                                                   128
        )

        self.FC_2_2 = nn.Sequential(

            # nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),

            nn.Linear(1300, 2048),  # 输出的维度 1                                                     1300
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 输出的维度 1                                                       1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 输出的维度 1                                                         768#
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 2                                                             512
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1600),  # 400到600 3                                                          768#res1
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 2048),  # 400到600 3                                                         768#
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 400到600 4                                                          1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 5                                                           768#res2
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 6                                                             512#
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 7                                                            768
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 800),  # 400到600 8                                                           512#res3
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 600),  # 400到600 9                                                            384
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 400),  # 400到600 10                                                           256
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 1)  # 600到1 11                                                                   128
        )
        '''
            self.FC_2_1 = nn.Sequential(
                #  TODO change input dimension
                nn.Linear(10 * opt.EMBEDDING_DIM, 400),
                nn.BatchNorm1d(400),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Linear(400, 600),
                nn.BatchNorm1d(600),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_2)
            )
        self.FC_2_2 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_2)
        )
        '''
        self.FC_3_1 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(1300, 2048),  # 输出的维度 1                                                     1300
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 输出的维度 1                                                       1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 输出的维度 1                                                         768#
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 2                                                             512
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1600),  # 400到600 3                                                          768#res1
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 2048),  # 400到600 3                                                         768#
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 400到600 4                                                          1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 5                                                           768#res2
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 6                                                             512#
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 7                                                            768
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 800),  # 400到600 8                                                           512#res3
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 600),  # 400到600 9                                                            384
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 400),  # 400到600 10                                                           256
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 1)  # 600到1 11
        )

        self.FC_3_2 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(1300, 2048),  # 输出的维度 1                                                     1300
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 输出的维度 1                                                       1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 输出的维度 1                                                         768#
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 2                                                             512
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1600),  # 400到600 3                                                          768#res1
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 2048),  # 400到600 3                                                         768#
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1600),  # 400到600 4                                                          1024
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 5                                                           768#res2
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 1600),  # 400到600 6                                                             512#
            nn.BatchNorm1d(1600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1600, 1000),  # 400到600 7                                                            768
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000, 800),  # 400到600 8                                                           512#res3
            nn.BatchNorm1d(800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800, 600),  # 400到600 9                                                            384
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 400),  # 400到600 10                                                           256
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 1)  # 600到1 11
        )

        self.FC_4_1 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_4)
        )

        self.FC_4_2 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_4)
        )

    def forward1_C(self, x):
        origin=x
        output=self.Con(origin)
        output+=origin
        relu = nn.ReLU(inplace=+True)
        output = relu(output)
        return output

    def forward1_FC_1(self, x):
        origin = x
        output=self.FC_R_1(origin)
        output+=origin
        relu = nn.ReLU(inplace=True)
        output = relu(output)
        return output

    def forward1_FC_2(self, x):
        origin = x
        output=self.FC_R_2(origin)
        output+=origin
        relu = nn.ReLU(inplace=True)
        output = relu(output)
        return output


    def forward(self, x):

        '''
        embedding layers
        '''
        output_encoder_plat_form = self.encoder_plat_form(x[:,1].long())
        output_encoder_biz_type = self.encoder_biz_type(x[:,2].long())
        output_encoder_product_id = self.encoder_product_id(x[:,3].long())

        output_encoder_cate1_id = self.encoder_cate1_id(x[:,4].long())
        output_encoder_cate2_id = self.encoder_cate2_id(x[:,5].long())
        output_encoder_cate3_id = self.encoder_cate3_id(x[:,6].long())
        output_encoder_seller_uid = self.encoder_seller_uid(x[:,7].long())

        output_encoder_company_name = self.encoder_company_name(x[:,8].long())
        output_encoder_rvcr_prov_name = self.encoder_rvcr_prov_name(x[:,9].long())
        output_encoder_rvcr_city_name = self.encoder_rvcr_city_name(x[:,10].long())
        output_encoder_lgst_company = self.encoder_lgst_company(x[:,11].long())
        output_encoder_warehouse_id = self.encoder_warehouse_id(x[:,12].long())
        output_encoder_shipped_prov_id = self.encoder_shipped_prov_id(x[:,13].long())


        concat_encoder_output = torch.cat((output_encoder_plat_form, 
        output_encoder_biz_type, output_encoder_product_id, 
        output_encoder_cate1_id, output_encoder_cate2_id,
        output_encoder_cate3_id, output_encoder_seller_uid,
        output_encoder_company_name, output_encoder_rvcr_prov_name,
        output_encoder_rvcr_city_name, output_encoder_lgst_company,
        output_encoder_warehouse_id, output_encoder_shipped_prov_id
        ), 1)#把10个embedding后的数据拼接起来

        '''
        Fully Connected layers
        you can attempt muti-task through uncommenting the following code and modifying related code in train()
        '''
        output_FC_0_1 = self.FC_0_1(concat_encoder_output)
        # output_FC_2_1 = self.FC_2_1(concat_encoder_output)
        # output_FC_3_1 = self.FC_3_1(concat_encoder_output)
        # output_FC_4_1 = self.FC_4_1(concat_encoder_output)

        output_FC_0_2 = self.FC_0_1(concat_encoder_output)
        # output_FC_2_2 = self.FC_2_1(concat_encoder_output)
        # output_FC_3_2 = self.FC_3_1(concat_encoder_output)
        # output_FC_4_2 = self.FC_4_1(concat_encoder_output)

        #print(output_FC_0_1.size())

        #output_FC_0_1 = self.FC_0_1(concat_encoder_output).reshape([concat_encoder_output.size()[0], 1, 13, 100])
        #output_FC_0_2 = self.FC_0_1(concat_encoder_output).reshape([concat_encoder_output.size()[0], 1, 13, 100])

        #output_FC_1_1 = self.Con1(output_FC_0_1)
        #output_FC_1_2 = self.Con1(output_FC_0_2)

        #output_FC_1_1 = self.forward1_C(output_FC_0_1)
        #output_FC_1_2 = self.forward1_C(output_FC_0_2)
        #output_FC_1_1 = self.forward1_C(output_FC_1_1)
        #output_FC_1_2 = self.forward1_C(output_FC_1_2)
        #output_FC_1_1 = self.forward1_C(output_FC_1_1)
        #output_FC_1_2 = self.forward1_C(output_FC_1_2)
        #output_FC_1_1 = self.forward1_C(output_FC_1_1)
        #output_FC_1_2 = self.forward1_C(output_FC_1_2)
        #output_FC_1_1 = self.Con2(output_FC_1_1).reshape([output_FC_0_1.size()[0], -1])
        #output_FC_1_2 = self.Con2(output_FC_1_2).reshape([output_FC_0_2.size()[0], -1])



        output_FC_1_1 = self.FC_1_1(output_FC_0_1)
        output_FC_1_2 = self.FC_1_2(output_FC_0_2)
        output_FC_2_1 = self.FC_2_1(output_FC_0_2)
        output_FC_2_2 = self.FC_2_2(output_FC_0_2)
        #output_FC_3_1 = self.FC_3_1(output_FC_0_2)
        #output_FC_3_2 = self.FC_1_2(output_FC_0_2)

        #output_FC_1_1 = self.FC_R_0(output_FC_0_1)
        #output_FC_1_2 = self.FC_R_0(output_FC_0_2)
        #output_FC_1_1 = self.forward1_FC_1(output_FC_1_1)
        #output_FC_1_2 = self.forward1_FC_1(output_FC_1_2)
        #output_FC_1_1 = self.forward1_FC_2(output_FC_1_1)
        #output_FC_1_2 = self.forward1_FC_2(output_FC_1_2)
        #output_FC_1_1 = self.FC_R_3(output_FC_1_1)
        #output_FC_1_2 = self.FC_R_3(output_FC_1_2)




        # return (output_FC_1_1, output_FC_2_1, output_FC_3_1, output_FC_4_1, output_FC_1_2, output_FC_2_2, output_FC_3_2, output_FC_4_2)
        return (output_FC_1_1, output_FC_1_2,output_FC_2_1,output_FC_2_2)
