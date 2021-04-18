import datetime
import numpy as np



ANSWER_FILE='sample_result.txt'
#TEST_FILE='test_1569417205.txt'均方根误差为22.224498194559985
'''
ACC:0.9448
均方根误差为21.64956004171909
ACC:0.9448
'''
'''
24 41.91 0.958 24.606 0.99711
25 43.69 0.969 26.4936 0.99818
372756 44.2154 0.977

'''



TEST_FILE='test_1570407531.txt'#1569588863 to tijiao#刚刚提交的50结果
def readDate(name='sample_result.txt'):
    ans=[]
    with open(name,'r') as f:
        for line in f.readlines():
            ans.append(datetime.datetime.strptime(line.strip(),'%Y-%m-%d %H'))
    return ans

ans=np.array(readDate(ANSWER_FILE))
test=np.array(readDate(TEST_FILE))

import pickle

hour_prob=[]
with open('hour_gailv.pkl','rb') as f:
    hour_prob=pickle.load(f)

with open('hour_index.pkl','rb') as f:
    hours=pickle.load(f)


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
calc(test)
#修改小时
for i in range(len(test)):
    if test[i].hour<=8:
        #print(0)
        hour=np.random.choice(hours,p=hour_prob)
        #hour = np.random.choice([13, 14, 15, 16], p=[0.4, 0.3, 0.2, 0.1])
        test[i]=test[i].replace(hour=hour)
        #test[i]=test[i]+datetime.timedelta(days=-1)
calc(test)


def save(test):
    for i in range(len(test)):
        test[i]=datetime.datetime.strftime(test[i],'%Y-%m-%d %H')
    with open('ansNew9-25_ZJZ.txt','w',encoding='utf-8') as f:
        for i in test:
            f.write(i+'\n')

save(test)


#save(test)

#修改日期
'''
for i in range(len(test)):
    if np.random.random()>0.8:
        #print(0)
        #day=int(np.random.choice([0,1],p=[0.2,0.8]))
        day = int(np.random.choice([0, -1], p=[0.4, 0.6]))
        #hour=np.random.choice([13,14,15,16],p=[0.4,0.3,0.2,0.1])
        test[i]=test[i]+datetime.timedelta(days=day)

'''
#calc(test)

#test=np.array(readDate('ansNew9-22_ZJZ.txt'))




