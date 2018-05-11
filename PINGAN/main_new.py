# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.linear_model import Ridge #LinearRegression,Ridge,Lasso,ElasticNet
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
import lightgbm as lgb

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

def tripmean(df):
    res = list()
    for i in df['TRIP_ID'].unique():
        res.append(df[df['TRIP_ID']==i]['SPEED'].sum()/60)
    return np.mean(res)

def dir(df):
    dirlst = df['DIRECTION'].tolist()
    result = 0
    if len(dirlst) >= 2:
        res = np.array(dirlst[1:])-np.array(dirlst[:-1])
        for i in range(res.size):
            if res[i]>180:
                res[i] = 360-abs(res[i])
        result = np.mean(abs(res))
    return result

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
    # print(df.dtypes)
    # print('train data:', df.shape)
    # lenlst = df.groupby('TERMINALNO')['TERMINALNO'].count()
    # print('one people(max):', lenlst.max(), 'average:', lenlst.mean(), 'std:',
    #       lenlst.std(), 'population', lenlst.count())
    # del lenlst
    df_person = pd.DataFrame()
    df_person['TERMINALNO'] = df['TERMINALNO'].unique()
    # print('get ID')
    # df_person.set_index('TERMINALNO')
    timeformat = pd.to_datetime(df.TIME.values, unit='s')
    df['month'] = timeformat.month
    df['hour'] = timeformat.hour
    df['week_Day'] = timeformat.weekday
    del timeformat
    df = df.drop(['TIME','TRIP_ID','LONGITUDE','LATITUDE','SPEED','HEIGHT','DIRECTION','CALLSTATE'], axis=1)
    # print('time transfer')
    df_person['count'] = df.groupby('TERMINALNO')['TERMINALNO'].count().tolist()
    # print('count')
    for i in range(12):
        df_person['mon'+str(i+1)] = df.groupby('TERMINALNO')['TERMINALNO','month'].apply(countmon,n=i+1).tolist()
        df_person['mon' + str(i+1)] = df_person['mon' + str(i+1)]/df_person['count']
        # print('month',i)
    df = df.drop('month', axis=1)
    # print('get mon')
    for i in range(24):
        df_person['hour'+str(i)] = df.groupby('TERMINALNO')['TERMINALNO','hour'].apply(counthour,n=i).tolist()
        df_person['hour' + str(i)] = df_person['hour' + str(i)]/df_person['count']
    df = df.drop('hour', axis=1)
    # print('get hour')
    del df
    df = readcsv(path_df)
    timeformat = pd.to_datetime(df.TIME.values, unit='s')
    df['week_Day'] = timeformat.weekday
    del timeformat
    df = df.drop(['TIME','TRIP_ID','LONGITUDE','LATITUDE','SPEED','HEIGHT','DIRECTION','CALLSTATE'], axis=1)
    for i in range(7):
        df_person['week_Day'+str(i)] = df.groupby('TERMINALNO')['TERMINALNO','week_Day'].apply(countwd,n=i).tolist()
        df_person['week_Day' + str(i)] = df_person['week_Day' + str(i)]/df_person['count']
        # print('week_Day', i)
    df = df.drop('week_Day', axis=1)
    del df
    df = readcsv(path_df)
    df = df.drop('TIME', axis=1)
    # print('get weekday')
    df_person['TRIP_ID'] = df.groupby('TERMINALNO')['TRIP_ID'].max().tolist()
    df_person['TRIP_MAX'] = df.groupby('TERMINALNO')['TERMINALNO','TRIP_ID','SPEED'].apply(tripmax).tolist()
    df_person['TRIP_MEAN'] = df.groupby('TERMINALNO')['TERMINALNO','TRIP_ID','SPEED'].apply(tripmean).tolist()
    df = df.drop('TRIP_ID', axis=1)
    df_person['LONGITUDE'] = df.groupby('TERMINALNO')['LONGITUDE'].mean().tolist()
    df = df.drop('LONGITUDE', axis=1)
    df_person['LATITUDE'] = df.groupby('TERMINALNO')['LATITUDE'].mean().tolist()
    df = df.drop('LATITUDE', axis=1)
    df_person['HEIGHT'] = df.groupby('TERMINALNO')['HEIGHT'].mean().tolist()
    df_person['HEIGHT_std'] = df.groupby('TERMINALNO')['HEIGHT'].std().tolist()
    df_person['HEIGHT_MAX'] = df.groupby('TERMINALNO')['HEIGHT'].max().tolist()
    df_person['HEIGHT_MIN'] = df.groupby('TERMINALNO')['HEIGHT'].min().tolist()
    df_person['HEIGHT_std'] = df_person['HEIGHT_std'].fillna(0)
    df = df.drop('HEIGHT', axis=1)
    df_person['SPEED'] = df.groupby('TERMINALNO')['SPEED'].mean().tolist()
    df_person['SPEED_std'] = df.groupby('TERMINALNO')['SPEED'].std().tolist()
    df_person['SPEED_MAX'] = df.groupby('TERMINALNO')['SPEED'].max().tolist()
    # df_person['SPEED_MIN'] = df.groupby('TERMINALNO')['SPEED'].min().tolist()
    df_person['SPEED_std'] = df_person['SPEED_std'].fillna(0)
    df = df.drop('SPEED', axis=1)
    # df_person['DIRECTION'] = df.groupby('TERMINALNO')['DIRECTION'].std().tolist()
    # df_person['DIRECTION'] = df_person['DIRECTION'].fillna(0)
    df_person['DIR'] = df.groupby('TERMINALNO')['TERMINALNO','DIRECTION'].apply(dir).tolist()
    df = df.drop('DIRECTION', axis=1)
    # print('get others')
    for i in range(5):
        df_person['CALLSTATE'+str(i)] = df.groupby('TERMINALNO')['TERMINALNO','CALLSTATE'].apply(countcs,n=i).tolist()
        df_person['CALLSTATE' + str(i)] = df_person['CALLSTATE' + str(i)]/df_person['count']
    df = df.drop('CALLSTATE', axis=1)
    # print('callstate')
    if len(df.columns)==2:
        df_person['Y'] = df.groupby('TERMINALNO')['Y'].mean().tolist()
    # print('done')
    del df
    df_person = df_person.drop('count',axis=1)
    return df_person

def train_predict(df_train,df_test):
    #    y_train=ss_y.fit_transform(y_train.reshape(-1,1))
    #    y_test=ss_y.transform(y_test.reshape(-1,1))
    #     GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
    #                                     max_depth=4, max_features='sqrt',
    #                                     min_samples_leaf=15, min_samples_split=10,
    #                                     loss='huber', random_state =5)
    y_min = 0
    y_max = df_train.iloc[:,-1].max()

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
    # enc = OneHotEncoder()
    # trainenc = enc.fit_transform(np.array(df_train['POS']).reshape(-1,1)).toarray()
    # testenc = enc.transform(np.array(df_test['POS']).reshape(-1,1)).toarray()
    df_train = df_train.drop('POS',axis=1)
    df_test = df_test.drop('POS', axis=1)

    # forest_range = [100, 300, 500]
    # parameters_forest = [{'n_estimators': forest_range, 'max_depth': [3]}]
    # gs_svr = GridSearchCV(estimator=RandomForestRegressor(criterion='mse', random_state=1), param_grid=parameters_forest, scoring='neg_mean_squared_error',
    #                       cv=5,n_jobs=-1)
    # gs_svr.fit(df_train.iloc[:,1:-1], df_train.iloc[:,-1])

    # xgb_range = [100, 300, 500]
    # parameters_xgb = [{'n_estimators': xgb_range, 'max_depth': [3], 'gamma': [0.8]}]
    # gs_svr = GridSearchCV(estimator=xgb.XGBRegressor(learning_rate=0.1, silent=1),
    #                       param_grid=parameters_xgb, scoring='neg_mean_squared_error',
    #                       cv=5, n_jobs=-1)
    # gs_svr.fit(df_train.iloc[:, 1:-1], df_train.iloc[:, -1])

    # GBoost = gs_svr.best_estimator_
    # print(np.sqrt(-gs_svr.best_score_),gs_svr.best_params_)
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

    # params = {
    #     "objective": 'reg:linear',
    #     "eval_metric": 'rmse',
    #     "seed": 1123,
    #     "booster": "gbtree",
    #     "min_child_weight": 5,
    #     "gamma": 0.1,
    #     "max_depth": 3,
    #     "eta": 0.009,
    #     "silent": 1,
    #     "subsample": 0.65,
    #     "colsample_bytree": .35,
    #     "scale_pos_weight": 0.9
    #     # "nthread":16
    # }

    # df_train = xgb.DMatrix(df_train.iloc[:, :-1], df_train.iloc[:, -1])
    # GBoost = xgb.train(params, df_train, num_boost_round=800)

    # test = xgb.DMatrix(df_test.iloc[:, :])
    # y_pred = GBoost.predict(test)

    # 采用lgb回归预测模型，具体参数设置如下
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.01, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
    # 训练、预测
    model_lgb.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
    y_pred = model_lgb.predict(df_test.iloc[:, :])

    for i in range(len(y_pred)):
        if y_pred[i]<0:
            y_pred[i] = y_min
        elif y_pred[i]>y_max:
            y_pred[i] = y_max
    sub_df = pd.DataFrame()
    sub_df['Id'] = df_test.iloc[:,0]
    sub_df['Pred'] = y_pred
    # sub_df = sub_df.groupby(['Id'])['Pred'].mean()
    # sub_df = pd.DataFrame(sub_df).reset_index()
    # sub_df['Pred'] = np.round(sub_df['Pred'], 3)
    sub_df.to_csv('model/test.csv', index=False)

def featureproc(df_train,df_test):
    namelst = df_train.columns[:-1]
    for i in namelst:
        if len(df_train[i].unique())==1 and len(df_test[i].unique())==1:
            df_train = df_train.drop(i,axis=1)
            df_test = df_test.drop(i,axis=1)
    return df_train,df_test

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

    # train_df,test_df = read_csv(path_train,path_test)
    # print('train data:',train_df.shape,'test data:',test_df.shape)
    # lenlst = train_df.groupby('TERMINALNO')['TERMINALNO'].count()
    # print('one people(max):',lenlst.max(),'average:',lenlst.mean(),'std:',
    #       lenlst.std(),'population',lenlst.count())
    print('****************** train data preprocess ******************')
    train_df_per = preprocess(path_train)
    print('****************** test data preprocess ******************')
    # del train_df
    test_df_per = preprocess(path_test)
    train_df_per,test_df_per = featureproc(train_df_per,test_df_per)
    print('****************** model preprocess ******************')
    # del test_df
    # prtlst = ['HEIGHT_std','SPEED_std','DIRECTION']
    # for i in prtlst:
    #     print('%.2f,%.2f' % (train_df_per[i].max(),train_df_per[i].min()),train_df_per[i].isnull().any(), np.isfinite(train_df_per[i].all()))
    # for i in range(len(test_df_per.columns)):
    #     print('%.2f,%.2f' % (test_df_per.iloc[:, i].max(), test_df_per.iloc[:, i].min()))
    train_predict(train_df_per,test_df_per)
    # train(train_df,test_df)
    # pre_df = predict(test_df)
    print('******************  end  **********************')
    # process()
