# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

path_train = "atec_anti_fraud_train.csv"  # 训练文件
path_test = "atec_anti_fraud_test_a.csv"  # 测试文件
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

traindata = pd.read_csv(path_train)
df=traindata[:int(traindata.shape[0]/8)]
df.to_csv('train.csv',index=False)
testdata = pd.read_csv(path_test)
traindata = traindata[traindata['label']!=-1]

labelneg = (traindata[traindata['label']!=-1].label.count() / len(traindata)) * 100

all_data_na = (traindata.isnull().sum() / len(traindata)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data1 = pd.DataFrame({'Missing Ratio' :all_data_na})

all_data_na = (testdata.isnull().sum() / len(testdata)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data2 = pd.DataFrame({'Missing Ratio' :all_data_na})

all_data = pd.concat((traindata, testdata))
all_data = all_data.drop('label',axis=1)

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data3 = pd.DataFrame({'Missing Ratio' :all_data_na})