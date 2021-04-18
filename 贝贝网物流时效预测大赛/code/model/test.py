import torch
from config import Config
import time
from dataLoader import TestSet
from evaluation import calculateAllMetrics
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from tqdm import tqdm
import os

'''
do inference on TestSet and output txt file
'''

time_delta = {}

def get_test_result():
    opt = Config()

    # load model
    if os.path.exists(opt.MODEL_SAVE_PATH):
        net = torch.load(opt.MODEL_SAVE_PATH1)
        print("==> load model successfully")
    else:
        print("==> model file dose not exist : ", opt.MODEL_SAVE_PATH)
        return

    # prepare test data
    print("==> loading data...")
    testset = TestSet(opt.TEST_FILE, opt=opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False)
    print("==> load data successfully")

    # do test
    net.eval()
    pred_signed_time = []
    for batch_idx, (inputs, payed_time) in enumerate(tqdm(testloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()

        inputs = torch.autograd.Variable(inputs)

        (output_FC_1_1, output_FC_1_2) = net(inputs.float())
        #  outputs = net(inputs.float()) # outputs.size: 800 * val_batchsize
        
        # calculate pred_signed_time via output
        for i in range(len(inputs)):
            temp_payed_time = payed_time[i]
            temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")
            # temp_pred_signed_time = temp_payed_time + relativedelta(hours = pred_time_interval)

            pred_time_day = output_FC_1_1[i]
            pred_time_hour = output_FC_1_2[i]
            pred_day_int = int(pred_time_day)
            if pred_day_int > 5:
                pred_day_int = 5
            if (int(inputs[i][4]) == 22 and int(inputs[i][5]) == 162):
                #print('haha')
                pred_day_int = pred_day_int - 1
            #temp_pred_signed_time = temp_payed_time + relativedelta(days = int(pred_time_day))
            if pred_day_int >=4:
                #pred_day_int-=1
                pred_day_int = 3
            temp_pred_signed_time = temp_payed_time + relativedelta(days=pred_day_int)
            '''
            if (int(pred_time_day)) not in time_delta:
                time_delta[int(pred_time_day)] = 1
            else:
                time_delta[int(pred_time_day)] += 1
                {2: 58252, 4: 5140, 5: 768, 3: 23765, 1: 11200, 
                13: 4, 6: 285, 10: 62, 8: 132, 0: 6, 7: 245,
                 12: 16, 14: 9, 11: 36, 9: 35, 21: 1, 19: 4, 
                 17: 10, 15: 6, 20: 4, 16: 15, 18: 5}
            '''
            temp_pred_signed_time = temp_pred_signed_time.replace(hour = int(pred_time_hour)%24)    

            # temp_pred_signed_time = temp_payed_time + relativedelta(hours = pred_time_interval)
            pred_signed_time.append(temp_pred_signed_time.strftime('%Y-%m-%d %H'))

    print(time_delta)
    # save predict result to txt file
    with open(opt.TEST_OUTPUT_PATH, 'w') as f:
        for res in pred_signed_time:
            f.write(res + "\n")


if __name__=='__main__':
    get_test_result()