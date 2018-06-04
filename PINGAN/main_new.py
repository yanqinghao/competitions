# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.linear_model import Ridge #LinearRegression,Ridge,Lasso,ElasticNet
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# import xgboost as xgb
import lightgbm as lgb

def countmon(df,n=1):
    return df[df['month']==n]['TERMINALNO'].count()

def counthour(df,n=1):
    return df[df['hour']==n]['TERMINALNO'].count()

def countwd(df,n=1):
    return df[df['week_Day']==n]['TERMINALNO'].count()

def countcs(df,n=1):
    return df[df['CALLSTATE']==n]['TERMINALNO'].count()

def tripmax(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        res.append(df[df['TRIP_ID']==i]['SPEED'].sum()/60)
    return np.max(res)

def tripmin(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        res.append(df[df['TRIP_ID']==i]['SPEED'].sum()/60)
    return np.min(res)

def tripmean(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        res.append(df[df['TRIP_ID']==i]['SPEED'].sum()/60)
    return np.mean(res)

def triptimemax(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        res.append(df[df['TRIP_ID']==i]['SPEED'].count())
    return np.max(res)

def triptimemin(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        res.append(df[df['TRIP_ID']==i]['SPEED'].count())
    return np.min(res)

def triptimemean(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        res.append(df[df['TRIP_ID']==i]['SPEED'].count())
    return np.mean(res)

def dir(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        dirlst = df[df['TRIP_ID'] == i]['DIRECTION'].tolist()
        result = 0
        if len(dirlst) >= 2:
            res1 = np.array(dirlst[1:]) - np.array(dirlst[:-1])
            for i in range(res1.size):
                if res1[i] > 180:
                    res1[i] = 360 - abs(res1[i])
            result = np.mean(abs(res1))
        res.append(result)
    return np.mean(res)
    # dirlst = df['DIRECTION'].tolist()
    # result = 0
    # if len(dirlst) >= 2:
    #     res = np.array(dirlst[1:])-np.array(dirlst[:-1])
    #     for i in range(res.size):
    #         if res[i]>180:
    #             res[i] = 360-abs(res[i])
    #     result = np.mean(abs(res))
    # return result

def dirmax(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        dirlst = df[df['TRIP_ID'] == i]['DIRECTION'].tolist()
        result = 0
        if len(dirlst) >= 2:
            res1 = np.array(dirlst[1:]) - np.array(dirlst[:-1])
            for i in range(res1.size):
                if res1[i] > 180:
                    res1[i] = 360 - abs(res1[i])
            result = np.max(abs(res1))
        res.append(result)
    return np.max(res)

def hightdmean(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        hlst = df[df['TRIP_ID']==i]['HEIGHT'].tolist()
        result = 0
        if len(hlst) >= 2:
            res1 = np.array(hlst[1:])-np.array(hlst[:-1])
            result = np.mean(abs(res1))
        res.append(result)
    return np.mean(res)

def hightdmax(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        hlst = df[df['TRIP_ID']==i]['HEIGHT'].tolist()
        result = 0
        if len(hlst) >= 2:
            res1 = np.array(hlst[1:])-np.array(hlst[:-1])
            result = np.max(abs(res1))
        res.append(result)
    return np.max(res)

def speeddmean(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        hlst = df[df['TRIP_ID']==i]['SPEED'].tolist()
        result = 0
        if len(hlst) >= 2:
            res1 = np.array(hlst[1:])-np.array(hlst[:-1])
            result = np.mean(abs(res1))
        res.append(result)
    return np.mean(res)

def speeddmax(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        hlst = df[df['TRIP_ID']==i]['SPEED'].tolist()
        result = 0
        if len(hlst) >= 2:
            res1 = np.array(hlst[1:])-np.array(hlst[:-1])
            result = np.max(abs(res1))
        res.append(result)
    return np.max(res)

def readcsv(path_df):
    df = pd.read_csv(path_df)
    # print(df.dtypes)
    if df.shape[1] == 10:
        df.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                      "CALLSTATE", "Y"]
        df[['DIRECTION', 'SPEED']] = df[['DIRECTION', 'SPEED']].replace(-1, np.nan)
        df[['DIRECTION', 'SPEED']] = df[['DIRECTION', 'SPEED']].fillna(method='ffill')
    else:
        df.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                      "CALLSTATE"]
        df[['DIRECTION', 'SPEED']] = df[['DIRECTION', 'SPEED']].replace('-1', np.nan)
        df[['DIRECTION', 'SPEED']] = df[['DIRECTION', 'SPEED']].fillna(method='ffill')
    df[['LONGITUDE', 'LATITUDE']] = df[['LONGITUDE', 'LATITUDE']].astype(np.float32)
    df[['HEIGHT', 'SPEED']] = df[['HEIGHT', 'SPEED']].astype(np.float32)
    df[['CALLSTATE']] = df[['CALLSTATE']].astype(np.int8)
    df[['DIRECTION']] = df[['DIRECTION']].astype(np.int16)
    df[['TRIP_ID']] = df[['TRIP_ID']].astype(np.int16)
    df[['TERMINALNO']] = df[['TERMINALNO']].astype(np.int32)
    return df

def preprocess(path_df):
    df = readcsv(path_df)
    df_person = pd.DataFrame()
    df_person['TERMINALNO'] = df['TERMINALNO'].unique()
    timeformat = pd.to_datetime(df.TIME.values, unit='s')
    df['month'] = timeformat.month
    df['hour'] = timeformat.hour
    df['week_Day'] = timeformat.weekday
    del timeformat
    df = df.drop(['TIME','TRIP_ID','LONGITUDE','LATITUDE','SPEED','HEIGHT','DIRECTION','CALLSTATE'], axis=1)
    #单用户计数
    df_person['count'] = df.groupby('TERMINALNO')['TERMINALNO'].count().tolist()
    #月频统计
    for i in range(12):
        df_person['mon'+str(i+1)] = df.groupby('TERMINALNO')['TERMINALNO','month'].apply(countmon,n=i+1).tolist()
        df_person['mon' + str(i+1)] = df_person['mon' + str(i+1)]/df_person['count']
    df = df.drop('month', axis=1)
    #时频统计
    for i in range(24):
        df_person['hour'+str(i)] = df.groupby('TERMINALNO')['TERMINALNO','hour'].apply(counthour,n=i).tolist()
        df_person['hour' + str(i)] = df_person['hour' + str(i)]/df_person['count']
    df = df.drop('hour', axis=1)
    #释放内存
    del df
    df = readcsv(path_df)
    timeformat = pd.to_datetime(df.TIME.values, unit='s')
    df['week_Day'] = timeformat.weekday
    del timeformat
    df = df.drop(['TIME','TRIP_ID','LONGITUDE','LATITUDE','SPEED','HEIGHT','DIRECTION','CALLSTATE'], axis=1)
    #周频统计
    for i in range(7):
        df_person['week_Day'+str(i)] = df.groupby('TERMINALNO')['TERMINALNO','week_Day'].apply(countwd,n=i).tolist()
        df_person['week_Day' + str(i)] = df_person['week_Day' + str(i)]/df_person['count']
    df = df.drop('week_Day', axis=1)
    #释放内存
    del df
    df = readcsv(path_df)
    df = df.drop('TIME', axis=1)
    #特征整理
    #tripid相关特征
    df_person['TRIP_ID'] = df.groupby('TERMINALNO')['TRIP_ID'].max().tolist()
    df_person['TRIP_MIN'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'SPEED'].apply(tripmin).tolist()
    df_person['TRIP_MEAN'] = df.groupby('TERMINALNO')['TERMINALNO','TRIP_ID','SPEED'].apply(tripmean).tolist()
    df_person['TRIP_TIME'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'SPEED'].apply(triptimemean).tolist()
    df_person['TRIP_TIMEMAX'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'SPEED'].apply(triptimemax).tolist()
    df_person['TRIP_TIMEMIN'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'SPEED'].apply(triptimemin).tolist()
    # LONGITUDE相关特征
    df_person['LONGITUDE'] = df.groupby('TERMINALNO')['LONGITUDE'].mean().tolist()
    df = df.drop('LONGITUDE', axis=1)
    # LATITUDE相关特征
    df_person['LATITUDE'] = df.groupby('TERMINALNO')['LATITUDE'].mean().tolist()
    df = df.drop('LATITUDE', axis=1)
    # HEIGHT相关特征
    df_person['HEIGHT'] = df.groupby('TERMINALNO')['HEIGHT'].mean().tolist()
    df_person['HEIGHT_std'] = df.groupby('TERMINALNO')['HEIGHT'].std().tolist()
    df_person['HEIGHT_MAX'] = df.groupby('TERMINALNO')['HEIGHT'].max().tolist()
    df_person['HEIGHT_MIN'] = df.groupby('TERMINALNO')['HEIGHT'].min().tolist()
    df_person['HEIGHT_DMEAN'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'HEIGHT'].apply(hightdmean).tolist()
    df_person['HEIGHT_DMAX'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'HEIGHT'].apply(hightdmax).tolist()
    df_person['HEIGHT_std'] = df_person['HEIGHT_std'].fillna(0)
    df = df.drop('HEIGHT', axis=1)
    # SPEED相关特征
    df_person['SPEED'] = df.groupby('TERMINALNO')['SPEED'].mean().tolist()
    df_person['SPEED_std'] = df.groupby('TERMINALNO')['SPEED'].std().tolist()
    df_person['SPEED_MAX'] = df.groupby('TERMINALNO')['SPEED'].max().tolist()
    df_person['SPEED_DMEAN'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'SPEED'].apply(speeddmean).tolist()
    df_person['SPEED_DMAX'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'SPEED'].apply(speeddmax).tolist()
    df_person['SPEED_std'] = df_person['SPEED_std'].fillna(0)
    df = df.drop('SPEED', axis=1)
    # DIRECTION相关特征
    df_person['DIR'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID','DIRECTION'].apply(dir).tolist()
    df_person['DIRMAX'] = df.groupby('TERMINALNO')['TERMINALNO', 'TRIP_ID', 'DIRECTION'].apply(dirmax).tolist()
    df = df.drop('TRIP_ID', axis=1)
    df = df.drop('DIRECTION', axis=1)
    # CALLSTATE频率统计
    for i in range(5):
        df_person['CALLSTATE'+str(i)] = df.groupby('TERMINALNO')['TERMINALNO','CALLSTATE'].apply(countcs,n=i).tolist()
        df_person['CALLSTATE' + str(i)] = df_person['CALLSTATE' + str(i)]/df_person['count']
    df = df.drop('CALLSTATE', axis=1)
    #train test分情况处理
    if len(df.columns)==2:
        df_person['Y'] = df.groupby('TERMINALNO')['Y'].mean().tolist()
    del df
    df_person = df_person.drop('count',axis=1)
    return df_person

def train_predict(df_train,df_test):
    y_min = 0
    y_max = df_train.iloc[:,-1].max()

    #删除位置信息
    # km = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, random_state=0)
    # km.fit(df_train[['LONGITUDE','LATITUDE']].values)
    # train_pos = km.predict(df_train[['LONGITUDE','LATITUDE']].values)
    # test_pos = km.predict(df_test[['LONGITUDE', 'LATITUDE']].values)
    # df_train['LONGITUDE'] = train_pos
    # df_test['LONGITUDE'] = test_pos
    df_train = df_train.drop('LATITUDE',axis=1)
    df_test = df_test.drop('LATITUDE', axis=1)
    df_train.rename(columns={'LONGITUDE': 'POS'}, inplace=True)
    df_test.rename(columns={'LONGITUDE': 'POS'}, inplace=True)
    df_train = df_train.drop('POS',axis=1)
    df_test = df_test.drop('POS', axis=1)

    # 常规模型
    # GBoost = RandomForestRegressor(n_estimators=500, max_depth=3, criterion='mse', random_state=1)
    # GBoost = xgb.XGBRegressor(n_estimators=500, max_depth=3, gamma=0.8, learning_rate=0.1, silent=1)
    # GBoost = Ridge(alpha=0.1)
    # GBoost = Lasso(alpha=50)

    # GBoost.fit(df_train.iloc[:,:-1], df_train.iloc[:,-1])
    #
    # features = np.row_stack((df_train.columns[:-1], GBoost.feature_importances_))
    # imp_df = pd.DataFrame(columns=['Names', 'importances'], data=features.T)
    # sorted_df = imp_df.sort_values('importances', ascending=False)
    # print(list(sorted_df['Names'].values))
    #
    # y_pred = GBoost.predict(df_test.iloc[:,:])

    # #GBR
    # params = {
    #     "n_estimators": 520,
    #     "max_depth": 3,
    #     "loss": 'ls',
    #     "learning_rate": 0.01,
    #     "subsample": 0.65,
    #     "max_features": .25,
    #     "random_state": 1234,
    # }
    # GBoost = GradientBoostingRegressor(**params)
    # GBoost.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
    #
    # features = np.row_stack((df_train.columns[:-1], GBoost.feature_importances_))
    # imp_df = pd.DataFrame(columns=['Names', 'importances'], data=features.T)
    # sorted_df = imp_df.sort_values('importances', ascending=False)
    # print(list(sorted_df['Names'].values))
    #
    # y_pred1 = GBoost.predict(df_test.iloc[:, :])

    # #xgb模型
    # params = {
    #     "objective": 'reg:linear',
    #     "eval_metric": 'rmse',
    #     "seed": 1123,
    #     "booster": "gbtree",
    #     "min_child_weight": 5,
    #     "gamma": 0.1,
    #     "max_depth": 3,
    #     "eta": 0.01,
    #     "silent": 1,
    #     "subsample": 0.65,
    #     "colsample_bytree": .25,
    #     "scale_pos_weight": 0.9
    #     # "nthread":16
    # }
    # params = {
    #     "objective": 'reg:linear',
    #     "eval_metric": 'rmse',
    #     "seed": 1123,
    #     "booster": "gbtree",
    #     "min_child_weight": 5,
    #     "gamma": 0.1,
    #     "max_depth": 3,
    #     "eta": 0.01,
    #     "silent": 1,
    #     "subsample": 0.65,
    #     "colsample_bytree": .25,
    #     "scale_pos_weight": 0.9
    #     # "nthread":16
    # }
    #
    # df_train = xgb.DMatrix(df_train.iloc[:, :-1], df_train.iloc[:, -1])
    # GBoost = xgb.train(params, df_train, num_boost_round=800)
    #
    # test = xgb.DMatrix(df_test.iloc[:, :])
    # y_pred = GBoost.predict(test)

    # 采用lgb回归预测模型，具体参数设置如下
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=6,
                              learning_rate=0.01, n_estimators=510,
                              max_bin = 55, bagging_fraction = 0.75,
                              bagging_freq = 5, feature_fraction = 0.28,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)#max_bin = 55,num_iterations=300
    # 训练、预测
    model_lgb.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
    y_pred = model_lgb.predict(df_test.iloc[:, :])

    # y_pred = 0.8*y_pred1+0.2*y_pred2

    #限制输出
    for i in range(len(y_pred)):
        if y_pred[i]<0:
            y_pred[i] = y_min
        elif y_pred[i]>y_max:
            y_pred[i] = y_max
    sub_df = pd.DataFrame()
    sub_df['Id'] = df_test.iloc[:,0]
    sub_df['Pred'] = y_pred
    sub_df.to_csv('model/test.csv', index=False)

def featureproc(df_train,df_test):
    namelst = df_train.columns[:-1]
    for i in namelst:
        if len(df_train[i].unique())==1 and len(df_test[i].unique())==1:
            df_train = df_train.drop(i,axis=1)
            df_test = df_test.drop(i,axis=1)
    return df_train,df_test

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    path_train = "./data/dm/train.csv"  # 训练文件
    path_test = "./data/dm/test.csv"  # 测试文件
    path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

    print('****************** train data preprocess ******************')
    train_df_per = preprocess(path_train)
    print('****************** test data preprocess ******************')
    test_df_per = preprocess(path_test)
    train_df_per,test_df_per = featureproc(train_df_per,test_df_per)
    print('****************** model preprocess ******************')
    train_predict(train_df_per,test_df_per)
    print('******************  end  **********************')
