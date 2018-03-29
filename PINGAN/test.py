# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:16:24 2018

@author: lenovo
"""

import pandas as pd
import numpy as np
import time
from scipy.cluster.vq import kmeans2 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df=pd.read_csv('./data/dm/train.csv', header=None)
df.columns=["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED","CALLSTATE", "Y"]
df=df[df['Y']<10]

df_time=df.TIME.map(lambda x:time.localtime(x))
df['month']=df_time.map(lambda x:x.tm_mon)
df['hour']=df_time.map(lambda x:x.tm_hour)
df['week_Day']=df_time.map(lambda x:x.tm_wday)


pos=df[['LONGITUDE','LATITUDE']]
res, idx = kmeans2(np.array(pos), 10, iter=20, minit='points')
# =============================================================================
# distortions=[]
# for i in range(1,10):
#     km=KMeans(n_clusters=i,
#               init='k-means++',
#               n_init=10,
#               max_iter=300,
#               random_state=0)
#     km.fit(pos)
#     distortions.append(km.inertia_)
# 
# plt.plot(range(1,10),distortions,marker='o')
# plt.xlabel('number of clusters')
# plt.ylabel('Distortion')
# plt.show()
# =============================================================================
df['pos']=idx

for i,item in enumerate(df.DIRECTION):
    if item<0:
        df.DIRECTION[i]=0
after_direc=df.DIRECTION[1:].reset_index(drop=True)
front_direc=df.DIRECTION[:-1].reset_index(drop=True)
direction=abs(after_direc-front_direc)
direction[direction.shape[0]]=0
for i,item in enumerate(direction):
    if item>180:
        direction[i]=item-180

def direc(val):
    if val>=0 and val<45:
        val=1
    elif val>=45 and val<90:
        val=2
    elif val>=90 and val<135:
        val=3
    else:
        val=4
    return val
direction=direction.map(direc)        
df['DIRECTION']=direction.values

id=df['TERMINALNO']
y=df['Y']
X=df.drop(['TERMINALNO','TIME','TRIP_ID','LONGITUDE','LATITUDE','Y'],axis=1)

# =============================================================================
# import seaborn as sns
# sns.distplot(y)
# 
# plt.scatter(X['SPEED'], y, c = "blue", marker = "s")
# plt.xlabel("SPEED")
# plt.ylabel("y")
# plt.show()
# 
# =============================================================================

categorical_features=X[['CALLSTATE','month','hour','week_Day','pos','DIRECTION']]
numerical_features=X[['HEIGHT','SPEED']]

data_categorical = pd.get_dummies(categorical_features,drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Ridge, Lasso
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


ridge = Ridge(alpha=10)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0001, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
score = rmsle_cv(ridge)
print("\nridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    

    
               