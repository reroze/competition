import torch
from config import Config
from model import Network,My_MSE_loss
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from dataLoader import TrainSet,ValSet
from model import Network
from evaluation import calculateAllMetrics
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from tqdm import tqdm
import os


opt = Config()

# prepare dataset
print("==> loading data...")
trainset = TrainSet(opt.TRAIN_FILE, opt=opt)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.TRAIN_BATCH_SIZE, shuffle=True)#随机选取batch_size个数


valset = ValSet(opt.VAL_FILE, opt=opt)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.VAL_BATCH_SIZE, shuffle=False)
print("==> load data successfully")

# setup network
net = Network(opt)
if opt.USE_CUDA:
    print("==> using CUDA")
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True

# set criterion (loss function)
criterion_1 = torch.nn.MSELoss()
criterion_2 = My_MSE_loss()

# you can choose metric in [accuracy, MSE, RankScore]
highest_metrics = 100
times =0

writer = SummaryWriter(comment='model1')


def train(epoch):
    net.train()#训练模型
    print("train epoch:", epoch)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.get_lr(epoch))
    #训练一次
    for batch_idx, (inputs, targets_sign_day, targets_sign_hour, targets_ship_day, targets_ship_hour, targets_got_day, targets_got_hour, targets_dlved_day, targets_dlved_hour,targets_dlved_signed_day) in enumerate(tqdm(trainloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()
            targets_sign_day = targets_sign_day.cuda()
            targets_sign_hour = targets_sign_hour.cuda()
            targets_ship_day = targets_ship_day.cuda()
            targets_ship_hour = targets_ship_hour.cuda()
            targets_got_day = targets_got_day.cuda()
            targets_got_hour = targets_got_hour.cuda()
            targets_dlved_day = targets_dlved_day.cuda()
            targets_dlved_hour = targets_dlved_hour.cuda()
            targets_dlved_signed_day = targets_dlved_signed_day.cuda()

        inputs = torch.autograd.Variable(inputs)
        targets_sign_day = torch.autograd.Variable(targets_sign_day.float())
        targets_sign_hour = torch.autograd.Variable(targets_sign_hour.float())
        targets_ship_day = torch.autograd.Variable(targets_ship_day.float())
        targets_ship_hour = torch.autograd.Variable(targets_ship_hour.float())
        targets_got_day = torch.autograd.Variable(targets_got_day.float())
        targets_got_hour = torch.autograd.Variable(targets_got_hour.float())
        targets_dlved_day = torch.autograd.Variable(targets_dlved_day.float())
        targets_dlved_hour = torch.autograd.Variable(targets_dlved_hour.float())
        targets_dlved_signed_day = torch.autograd.Variable(targets_dlved_signed_day.float())

        optimizer.zero_grad()

        (output_FC_1_1, output_FC_1_2,output_FC_2_1,output_FC_2_2) = net(inputs.float())#直接向net中输入inputs

                
        output_FC_1_1 = output_FC_1_1.reshape(-1)#签收时间-支付时间
        # output_FC_2_1 = output_FC_2_1.reshape(-1)#发货时间-支付时间
        # output_FC_3_1 = output_FC_3_1.reshape(-1)
        # output_FC_4_1 = output_FC_4_1.reshape(-1)
        
        output_FC_1_2 = output_FC_1_2.reshape(-1)
        # output_FC_2_2 = output_FC_2_2.reshape(-1)
        # output_FC_3_2 = output_FC_3_2.reshape(-1)
        # output_FC_4_2 = output_FC_4_2.reshape(-1)

        loss_1_1 = criterion_2(output_FC_1_1, targets_sign_day)
        # loss_2_1 = criterion_1(output_FC_2_1, targets_ship_day)
        # loss_3_1 = criterion_1(output_FC_3_1, targets_got_day)
        # loss_4_1 = criterion_1(output_FC_4_1, targets_dlved_day)
        
        loss_1_2 = criterion_1(output_FC_1_2, targets_sign_hour)

        loss_1_3 = criterion_1(output_FC_2_1, targets_dlved_day)

        loss_1_4 = criterion_1(output_FC_2_2, targets_dlved_signed_day)
        # loss_2_2 = criterion_1(output_FC_2_2, targets_ship_hour)
        # loss_3_2 = criterion_1(output_FC_3_2, targets_got_hour)
        # loss_4_2 = criterion_1(output_FC_4_2, targets_dlved_hour)

        loss_day  = (loss_1_1+loss_1_3+loss_1_4)/2
        loss_hour = loss_1_2
        #loss_hour = loss_1_2

        loss = loss_day*24  + loss_hour
        loss.backward()
        writer.add_scalar('Train', loss, batch_idx+epoch*5023)

        '''
        model1:#24 64.5 0.971 #30:64.2 0.969(收敛)63.0 0.965(最优) 24要好一点 20 62.95 0.957 63.3 0.965 #61.50 0.951134
        '''

        #torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)
        optimizer.step()

        # TODO add to tensorboard
        if batch_idx == 1:#tqdm(trainloader) #每进行一次dataloder记一次数
            print("==> epoch {}: loss_day is {}, loss_hour is {} ".format(epoch, loss_day, loss_hour))
            #print(output_FC_1_2, targets_sign_hour)

def val(epoch):
    global highest_metrics
    net.eval()#生成模型
    print('hello')
    pred_signed_time = []
    real_signed_time = []
    for batch_idx, (inputs, payed_time, signed_time) in enumerate(tqdm(valloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()

        inputs = torch.autograd.Variable(inputs)
        (output_FC_1_1, output_FC_1_2,output_FC_2_1, output_FC_2_2) = net(inputs.float())
        #print('output_hour' ,output_FC_1_2)
        # calculate pred_signed_time via output
        for i in range(len(inputs)):

            pred_time_day = output_FC_1_1[i]#evaluate
            pred_dlved_day = output_FC_2_1[i]
            pred_dlved_signed_day = output_FC_2_2[i]

            pred_time_day = (pred_time_day+pred_dlved_day+pred_dlved_signed_day)/2
            pred_time_hour = output_FC_1_2[i]

            temp_payed_time = payed_time[i]
            temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")
            if(pred_time_hour == float('nan')):
                print('error_input', inputs)
            temp_payed_time = temp_payed_time.replace(hour = int(pred_time_hour)%24)

            temp_pred_signed_time = temp_payed_time + relativedelta(days = int(pred_time_day))
            temp_pred_signed_time = temp_pred_signed_time.replace(hour = int(pred_time_hour)%24)
            temp_pred_signed_time = temp_pred_signed_time.replace(minute = 0)
            temp_pred_signed_time = temp_pred_signed_time.replace(second = 0)
            # temp_pred_signed_time.

            pred_signed_time.append(temp_pred_signed_time.strftime("%Y-%m-%d %H"))
            real_signed_time.append(signed_time[i])

    (rankScore_result, onTimePercent_result, accuracy_result) = calculateAllMetrics(real_signed_time, pred_signed_time)
    print("==> epoch {}: rankScore is {}, onTimePercent is {}, accuracy is {}".format(epoch, rankScore_result, onTimePercent_result, accuracy_result))

    # save model
    if rankScore_result < highest_metrics:
        print("==> saving model")
        print("==> onTimePercent {} | rankScore {} ".format(onTimePercent_result, rankScore_result))
        highest_metrics = rankScore_result
        torch.save(net, opt.MODEL_SAVE_PATH)
        f = open(os.path.join(opt.MODEL_SAVE_FOLDER, 'record.txt'), 'w')
        f.write(str(epoch))
        f.close()
    if onTimePercent_result > opt.ontime:
        print("==> saving model")
        print("==> onTimePercent {} | rankScore {} ".format(onTimePercent_result, rankScore_result))
        opt.ontime = onTimePercent_result
        torch.save(net, opt.MODEL_SAVE_PATH1)
        f = open(os.path.join(opt.MODEL_SAVE_FOLDER, 'record2.txt'), 'w')
        f.write(str(epoch))
        f.close()



start = 0
# start training
if __name__=='__main__':
    if os.path.exists(opt.MODEL_SAVE_PATH):
        net = torch.load(opt.MODEL_SAVE_PATH)
        f = open(os.path.join(opt.MODEL_SAVE_FOLDER, 'record.txt'), 'r')
        context = f.readlines()
        start = int(context[0].strip('\r\n'))
        print('start', start)
        f.close()
    for i in range(start, opt.NUM_EPOCHS):
        times = 0
        train(i)#训练一次


        if i % opt.val_step == 0:
            val(i)
            print('times:', times)
