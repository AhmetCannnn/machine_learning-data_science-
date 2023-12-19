# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:53:36 2023

@author: AHMET CAN
"""

import pandas as pd

url = "http://bilkav.com/satislar.csv"

veriler = pd.read_csv(url)

X = veriler.iloc[:,0:1].values
Y = veriler.iloc[:,1].values

bolme = 0.33

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = bolme)    


import pickle

dosya = "model.kayıt"

'''
pickle.dump(lr,open(dosya,"wb")) / modelin verdiği sonucları kaydetme
'''

yüklenen = pickle.load(open(dosya,"rb")) #kaydettiğimiz sonucları tekrardan yükleyip kullanmak


print(yüklenen.predict(X_test))
