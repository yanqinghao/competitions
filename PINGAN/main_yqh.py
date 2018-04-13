# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score

path_train = "./data/dm/train.csv"  # 训练文件,更改路径提交时删去pingan
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    return tempdata


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    import numpy as np

    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                writer.writerow([item[0], np.random.rand()]) # 随机值
                
                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    rawdata = read_csv()
    train = rawdata
    timeformat = pd.to_datetime(train.TIME.values, unit='s')
    train['YEAR'] = timeformat.year
    train['MON'] = timeformat.month
    train['DAY'] = timeformat.day
    train['HOUR'] = timeformat.hour
    train['MIN'] = timeformat.minute
    train['SEC'] = timeformat.second
    train['WEEKDAY'] = timeformat.weekday
    ohscaler = OneHotEncoder()
    onehottrain = ohscaler.fit_transform(train['CALLSTATE'].values.reshape(-1, 1))
    ohdf = pd.DataFrame(columns=['UNKNOWN', 'CALLOUT', 'CALLIN', 'CONNECT', 'DISCONNECT'], data=onehottrain.toarray())
    train = pd.concat([train, ohdf],axis=1)
    train = train.drop(['TERMINALNO','TRIP_ID','TIME','YEAR','SEC','CALLSTATE'], axis=1)
    namecol = list(train.columns[:5]) + list(train.columns[6:])
    namecol.append(train.columns[5])
    train = train[namecol]
    print('******')
    forest = RandomForestRegressor(n_estimators=400, criterion='mse', random_state=1, n_jobs=-1)
    forest.fit(train.ix[:, :-1], train.ix[:, -1])
    features = np.row_stack((train.columns[:-1], forest.feature_importances_))
    imp_df = pd.DataFrame(columns=['Names', 'importances'], data=features.T)
    sorted_df = imp_df.sort_values('importances', ascending=False)
    namelst = list(sorted_df['Names'].values[0:10])
    namelst.append('Y')
    train = train.ix[:, namelst]

    elastic_range_1 = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01]
    elastic_range_2 = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    parameters_elastic = [{'alpha': elastic_range_1, 'l1_ratio': elastic_range_2}]
    gs_elastic = GridSearchCV(estimator=ElasticNet(), param_grid=parameters_elastic, scoring='neg_mean_squared_error',
                              cv=5, n_jobs=-1)
    gs_elastic.fit(train.ix[:, :-1], train.ix[:, -1])
    print(np.sqrt(-gs_elastic.best_score_))
    print(gs_elastic.best_params_)

    scores = cross_val_score(estimator=LinearRegression(),X=train.ix[:, :-1], y=train.ix[:, -1],scoring='neg_mean_squared_error',cv=5)
    print(np.sqrt(-scores.mean()))

    forest_range = range(400,1100,100)
    forest_range1 = range(2, 7, 1)
    parameters_forest = [{'n_estimators': [600], 'max_depth': forest_range1}]
    gs_forest = GridSearchCV(estimator=RandomForestRegressor(criterion='mse', random_state=1), param_grid=parameters_forest, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    gs_forest.fit(train.ix[:, :-1], train.ix[:, -1])
    print(np.sqrt(-gs_forest.best_score_))
    print(gs_forest.best_params_)

    # distortions = []
    # for i in range(1, 11):
    #     km = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    #     km.fit(train[['LONGITUDE','LATITUDE']].values)
    #     distortions.append(km.inertia_)
    # plt.plot(range(1, 11), distortions, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Distortion')
    # plt.show()
    a = 1
    # process()
