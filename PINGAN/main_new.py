# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def read_csv(path_train,path_test):
    """
    文件读取模块，头文件见columns.
    :return:
    """
    df_train = pd.read_csv(path_train)
    df_train.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    df_train[['DIRECTION', 'SPEED']] = df_train[['DIRECTION','SPEED']].replace(-1,np.nan)
    df_train[['DIRECTION', 'SPEED']] = df_train[['DIRECTION','SPEED']].fillna(method='ffill')
    df_test = pd.read_csv(path_test)
    df_test.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                       "CALLSTATE"]
    df_test[['DIRECTION', 'SPEED']] = df_test[['DIRECTION', 'SPEED']].replace('-1', np.nan)
    df_test[['DIRECTION', 'SPEED']] = df_test[['DIRECTION', 'SPEED']].fillna(method='ffill')
    return df_train,df_test

def preprocess():

    return

def train(df_train,df_test):
    ID_train = df_train['TERMINALNO']
    ID_test = df_test['TERMINALNO']
    ntrain = df_train.shape[0]
    ntest = df_test.shape[0]
    y_train = df_train['Y']
    Combined_data = pd.concat([df_train, df_test]).reset_index(drop=True)
    del df_train, df_test
    print('*****1')
    timeformat = pd.to_datetime(Combined_data.TIME.values, unit='s')
    print('*****2')
    Combined_data['month'] = timeformat.month
    print('*****3')
    Combined_data['hour'] = timeformat.hour
    print('*****4')
    Combined_data['week_Day'] = timeformat.weekday
    del timeformat
    print('*****5')

    # df_time=Combined_data.TIME.map(lambda x:time.localtime(x))
    # Combined_data['month']=df_time.map(lambda x:x.tm_mon)
    # Combined_data['hour']=df_time.map(lambda x:x.tm_hour)
    # Combined_data['week_Day']=df_time.map(lambda x:x.tm_wday)
    # pos=Combined_data[['LONGITUDE','LATITUDE']]
    print('*****6')
    # km = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, random_state=0)
    print('*****7')
    # km.fit(Combined_data[['LONGITUDE', 'LATITUDE']].values)
    # joblib.dump(km, 'km.pkl')
    # km1 = joblib.load('km.pkl')
    print('*****8')
    # res, idx = kmeans2(np.array(pos), 10, iter=20, minit='points')
    # Combined_data['pos']=km.predict(Combined_data[['LONGITUDE', 'LATITUDE']].values)
    print('*****9')
    Combined_data[Combined_data['DIRECTION'] < 0] = 0
    print('*****10')
    # for i,item in enumerate(Combined_data.DIRECTION):
    #     if item<0:
    #         Combined_data.DIRECTION[i]=0
    print('********11')
    after_direc = Combined_data.DIRECTION[1:].reset_index(drop=True)
    front_direc = Combined_data.DIRECTION[:-1].reset_index(drop=True)
    direction = abs(after_direc - front_direc)
    del after_direc, front_direc
    direction[direction.shape[0]] = 0
    for i, item in enumerate(direction):
        if item > 180:
            direction[i] = item - 180

    def direc(val):
        if val >= 0 and val < 45:
            val = 1
        elif val >= 45 and val < 90:
            val = 2
        elif val >= 90 and val < 135:
            val = 3
        else:
            val = 4
        return val

    direction = direction.map(direc)
    Combined_data['DIRECTION'] = direction.values
    del direction
    print('*******2')
    # X=Combined_data.drop(['TERMINALNO','TIME','TRIP_ID','LONGITUDE','LATITUDE','Y'],axis=1)
    # categorical_features=X[['CALLSTATE','month','hour','week_Day','DIRECTION']]
    # numerical_features=X[['HEIGHT','SPEED']]
    # # data_categorical = pd.get_dummies(categorical_features,drop_first=True)
    # Combined_data = pd.concat([categorical_features, numerical_features], axis = 1)
    Combined_data = Combined_data[['CALLSTATE', 'month', 'hour', 'week_Day', 'DIRECTION', 'HEIGHT', 'SPEED']]
    print('********3')
    X_train = Combined_data[:ntrain]
    X_test = Combined_data[ntrain:]
    del Combined_data
    X_test = X_test.reset_index(drop=True)
    print('********4')

    # from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import LinearRegression
    print('********41')
    # ss_X=RobustScaler()
    ss_X = StandardScaler()
    # ss_y=RobustScaler()
    print('********42')
    # X_train=ss_X.fit_transform(X_train)
    print('********43')
    # X_test=ss_X.transform(X_test)
    print('********5')
    #    y_train=ss_y.fit_transform(y_train.reshape(-1,1))
    #    y_test=ss_y.transform(y_test.reshape(-1,1))
    #     GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
    #                                     max_depth=4, max_features='sqrt',
    #                                     min_samples_leaf=15, min_samples_split=10,
    #                                     loss='huber', random_state =5)
    GBoost = LinearRegression()
    print('********6')
    GBoost.fit(X_train, y_train)
    del X_train, y_train
    print('********7')
    y_pred = GBoost.predict(X_test)
    del X_test
    print('********8')
    sub_df = pd.DataFrame()
    sub_df['Id'] = ID_test
    sub_df['Pred'] = y_pred
    sub_df = sub_df.groupby(['Id'])['Pred'].mean()
    sub_df = pd.DataFrame(sub_df).reset_index()
    sub_df['Pred'] = np.round(sub_df['Pred'], 3)
    sub_df.to_csv('model/test.csv', index=False)
    print('********9')


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
                writer.writerow([item[0], np.random.rand()])  # 随机值

                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重

def predict(test_df):
    return

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    path_train = "./data/dm/train.csv"  # 训练文件
    path_test = "./data/dm/test.csv"  # 测试文件
    path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

    train_df,test_df = read_csv(path_train,path_test)
    print('train data:',train_df.shape,'test data:',test_df.shape)
    lenlst = train_df.groupby('TERMINALNO')['TERMINALNO'].count()
    print('one people(max):',lenlst.max(),'average:',lenlst.mean(),'std:',
          lenlst.std(),'population',lenlst.count())
    print(train_df.head(2))
    print(test_df.head(2))
    # train(train_df,test_df)
    # pre_df = predict(test_df)

    print('******************  end  **********************')
    # process()
