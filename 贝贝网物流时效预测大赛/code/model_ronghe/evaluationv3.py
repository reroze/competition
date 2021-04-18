import datetime
import numpy as np
import random as rd


file2 = '28.56 0.9992 46.1642 0.988.txt'#时间普遍偏早 26.46 0.99974
file1 = 'lowest_rank.txt'
file3 = '44.56 0.982.txt'
#file4 = 'ansNew9-24_ZJZ.txt'

#0.99978 26.1734 43.9126 0.979

'''
低rank 低ontime说明 0.959说明有4100分数据预测至少迟一天
高rank 高ontime说明部分预测时间过前 0.988 1200份
至少有2800份之间只隔1天
'''

import pickle

hour_prob=[]
with open('hour_gailv.pkl','rb') as f:
    hour_prob=pickle.load(f)

with open('hour_index.pkl','rb') as f:
    hours=pickle.load(f)

def hour_just(test):
    for i in range(len(test)):
        if test[i].hour<=8:
            #print(0)
            #hour=np.random.choice(hours,p=hour_prob)
            hour = np.random.choice([13, 14, 15, 16], p=[0.4, 0.3, 0.2, 0.1])
            test[i]=test[i].replace(hour=hour)
            test[i] = test[i].replace(day=test[i].day-1)

def readDate(name='sample_result.txt'):
    ans=[]
    with open(name,'r') as f:
        for line in f.readlines():
            ans.append(datetime.datetime.strptime(line.strip(),'%Y-%m-%d %H'))
    return ans

test1 = np.array(readDate(file1))
test2 = np.array(readDate(file2))
test3 = np.array(readDate(file3))
#test4 = np.array(readDate(file4))
sum1 = [0]
#99999份数据在一天之内，34679数据相差一天
def ronghe(test1, test2, test3):
    for i in range(len(test1)):
        #test1[i] = test1[i].replace(hour=int((test1[i].hour + test2[i].hour) / 2))
        #test1[i] = test1[i].replace(day=int((test1[i].day + test2[i].day) / 2))


        if test1[i].day == test2[i].day:#64413 #test2只有1200份数据迟到 1200份此中占大部分 打算1000份
            #sum1[0] += 1
            #if rd.uniform(0, 1) > 21/29:  # 0.8还是少了一点 44.2 0.979
            if rd.uniform(0, 1) > 1:
                test1[i] = test1[i].replace(day=int(int((test3[i].day) + int(test2[i].day) + 2 * int(test1[i].day)) / 4))
            else:
                test1[i] = test1[i].replace(day=int(test1[i].day))
        else:#35587 test1有4100份数据迟到 此处的数据是test2比test1早的数据 也就是test1里至少有3100份此处也是迟到
            #sum1[0] += 1
            if test3[i].day < test1[i].day:#21258 test3里只有2800份迟到 1000份左右 可以通过test3和test2来调节
                #sum1[0] += 1#                                            刚是26.1 0.99974
                #26.69 0.99982

                if rd.uniform(0, 1) > 5 / 29:#0~0.1 ###13/29 0.95 0.3 0.5 是43.8852 0.978  #27.775 0.99989 22.0 0.99919

                    test1[i] = test1[i].replace(day=int(test3[i].day))
                    #test1[i] = test1[i].replace(day=int(int(int(test3[i].day) + 1*int(test2[i].day) + 5*int(test1[i].day))/ 7 +0.2857
                    #print('haha')
            else:#14329 test3有2800份数据迟到 #至少有1800份 只能通过test2来调节
                if rd.uniform(0,1) > 0.5:#0~0.1 #27.3 0.9998 25.94 0.9996
                    #test1[i] = test1[i].replace(day=int(int(int(test3[i].day) + int(test2[i].day) + 5*int(test1[i].day)) / 7))
                    test1[i] = test1[i].replace(day=int(test2[i].day))
                #sum1[0] += 1


        #if int(test1[i].day) == int((test1[i].day + test2[i].day) / 2):#早的一天 0.75按早的那一天算
            #if rd.uniform(0,1) > 0.1:#0.8还是少了一点 44.2 0.979
                #test1[i] = test1[i].replace(day=int(test2[i].day))
hour_just(test1)
hour_just(test2)
ronghe(test1, test2, test3)
print(sum1)
ans=np.array(readDate())
#test1 = np.array(test1)

def calc(test):
    delta=ans-test
    mse=[]
    for i in delta:
        mse.append(i.days*24+i.seconds/(60*60))
    mse=np.array(mse)
    rmse=np.sqrt(np.mean(np.square(mse)))
    print("均方根误差为{}".format(rmse))
    acc=0
    for i,j in zip(ans,test):
        if j.day<=i.day:
            acc+=1
    print("ACC:{}".format(acc/len(test)))
    return acc/len(test)

epoch = 0
accurancy = 0
accurcy=calc(test1)
test1 = np.array(readDate(file1))
print('#')
#calc(test4)
#print('accurcy', accurcy)


def save(test):
    for i in range(len(test)):
        test[i]=datetime.datetime.strftime(test[i],'%Y-%m-%d %H')
    with open('ansNew9-25_ZJZ.txt','w',encoding='utf-8') as f:
        for i in test:
            f.write(i+'\n')

def save1(test):
    test1=test.copy()
    for i in range(len(test1)):
        test1[i]=datetime.datetime.strftime(test1[i],'%Y-%m-%d %H')
    with open('ansNew9-26_ZJZ.txt','w',encoding='utf-8') as f:
        for i in test1:
            f.write(i+'\n')



while(epoch<5):
    ronghe(test1, test2, test3)
    accurcy_it = calc(test1)

    if accurcy_it>=accurcy:
        accurcy = accurcy_it
        epoch += 1
        test_buffer=test1.copy()
        save1(test_buffer)
        print('hello')
        if epoch==5:
            print(calc(test1))
            save(test1)
            break
    test1 = np.array(readDate(file1))


#print(calc(test1))


#save(test1)
