# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:10:33 2023

@author: AHMET CAN
"""

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme

veriler = pd.read_csv('satıs_verileri.csv')
#pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme
aylar = veriler[["Aylar"]]
print(aylar)
satislar = veriler[["Satislar"]]
print(satislar)

satislar2 = veriler.iloc[:,:1].values

print(satislar2)



#Veri kümesinin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

'''
Bu kısımda isteğe bağlı olarak veri ölçeklendirilebilir.

#öznitelik ölçekleme

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''

#model insası(linear regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")


































