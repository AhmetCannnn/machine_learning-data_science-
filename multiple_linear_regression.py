# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:53:48 2023

@author: AHMET CAN
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#veri yukleme

veriler = pd.read_csv('odev_tenis.csv') # You can find the 'odev_tenis' file in my repositories


print(veriler)


#play kolonundaki kategorik verileri sayısal değerlere çevirdik
play = veriler.iloc[:,-1:].values
print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

play[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(play)

#windy kolonundaki kategorik verileri sayısal değerlere çevirdik
windy = veriler.iloc[:,-2:-1].values
print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(windy)


# alternatif olarak bütün kolonlardaki verilerin toplu olarak sayısal değerlere çevrilmesi(toplu encode)
from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


#numpy dizileri dataframe dönüşümü
havadurumu = pd.DataFrame(data = c, index = range(14), columns=["o", "r", "s"])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]], axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


#Python ile geri eleme (Backkward Elimination) yöntemi

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler, axis=1)


X_l  = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype = float)
model = sm.OLS(sonveriler.iloc[:,-1:], X_l).fit()
print(model.summary())

sonveriler = sonveriler.iloc[:,1:]

#en yüksek p değerine sahip olan değeri eledik
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler, axis=1)


X_l  = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype = float)
model = sm.OLS(sonveriler.iloc[:,-1:], X_l).fit()
print(model.summary())

#ilk kolonu atarak sistemi tekrar eğittik
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#ve sonuçların bir miktar da olsa iyileştiğini gözlemledik

