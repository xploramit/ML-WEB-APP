# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:38:58 2022

@author: Amit
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Boston Dataset.csv")
df.head()
X = df.drop(['medv'],axis=1)
y = df['medv']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
lr = LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)
etr = ExtraTreesRegressor()
etr.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = dt.predict(X_test)
y_pred4 = rf.predict(X_test)
y_pred5 = etr.predict(X_test)
y_pred6 = gr.predict(X_test)

df1 = pd.DataFrame({'Actual':y_test,'LR':y_pred1, 'SVM':y_pred2,'DT':y_pred3,'RF':y_pred4,
                  'ETR':y_pred5, 'GR':y_pred6})
df1
from sklearn import metrics
score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)
score5 = metrics.r2_score(y_test,y_pred5)
score6 = metrics.r2_score(y_test,y_pred6)
print(score1,score2,score3,score4,score5,score6)
data1 = {'crim':0.00632,
    'zn':18.0,
    'indus':2.31,
    'chas':0,
    'nox':0.538,
    'rm':6.575,
    'age':65.2,
    'dis':4.0900,
    'rad':1,
    'tax':296,
    'ptratio':15.3,
    'black':396.90,
    'lstat':4.98,}
df = pd.DataFrame(data1,index=[0])
df
new_pred = gr.predict(df)
print("Price of House is : ",new_pred[0])
import joblib
gr = GradientBoostingRegressor()
gr.fit(X,y)
joblib.dump(gr,'model_joblib_gr2')
model = joblib.load('model_joblib_gr2')
model.predict(df)