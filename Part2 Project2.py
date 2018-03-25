# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:14:49 2017

@author: Sanyu
"""

# Project2 Part2

# In[] 
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# In[]
# read the dataset

Homeprice = pd.read_csv('C:\Anaconda\home_price.csv')
Homeprice.head()

# In[]
#coding for zipcodes
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
homeprice["Zip"] = lb_make.fit_transform(homeprice["zipcode"])

# In[]
#split in test and train dataset

X = Homeprice[['bedrooms', 'bathrooms','sqft_living','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_renovated','yr_built','lat','long','zipcode','sqft_living15','sqft_lot15']]
y = Homeprice['price']

# In[]
# scaling the parameter
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# we must apply the scaling to the test set that we computed for the training set
#X_test_scaled = scaler.transform(X_test)

# In[]
# all variables
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

TrainScores = []
TestScores = []

for i in range(2,4):
    poly = PolynomialFeatures(degree=i)
    X_F1_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    print('Degree = {:.2f}\nR-squared score (training): {:.3f}'
         .format(i,linreg.score(X_train, y_train)))
    print('(Degree = {:.2f}\nR-squared score (test): {:.3f}\n'
         .format(i,linreg.score(X_test, y_test)))
    trainsc = linreg.score(X_train,y_train)
    testsc = linreg.score(X_test, y_test)
    TrainScores.append(trainsc)
    TestScores.append(testsc)

# In[]

import matplotlib.pyplot as plt
plt.plot(TrainScores)
plt.plot(TestScores)
plt.show()

# In[]
# removing sqft_lot15 and sqft_living15
X = Homeprice[['bedrooms', 'bathrooms','sqft_living','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_renovated','yr_built','lat','long','zipcode']]
y = Homeprice['price']

# In[]

TrainScores1 = []
TestScores1 = []

for i in range(2,4):
    poly = PolynomialFeatures(degree=i)
    X_F1_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    print('Degree = {:.2f}\nR-squared score (training): {:.3f}'
         .format(i,linreg.score(X_train, y_train)))
    print('(Degree = {:.2f}\nR-squared score (test): {:.3f}\n'
         .format(i,linreg.score(X_test, y_test)))
    trainsc = linreg.score(X_train,y_train)
    testsc = linreg.score(X_test, y_test)
    TrainScores1.append(trainsc)
    TestScores1.append(testsc)

   
# In[]

import matplotlib.pyplot as plt
plt.plot(TrainScores1)
plt.plot(TestScores1)
plt.show()

# In[]
# removing long
X = Homeprice[['bedrooms', 'bathrooms','sqft_living','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_renovated','yr_built','lat','zipcode']]
y = Homeprice['price']

# In[]

TrainScores2 = []
TestScores2 = []

for i in range(2,4):
    poly = PolynomialFeatures(degree=i)
    X_F1_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    print('Degree = {:.2f}\nR-squared score (training): {:.3f}'
         .format(i,linreg.score(X_train, y_train)))
    print('(Degree = {:.2f}\nR-squared score (test): {:.3f}\n'
         .format(i,linreg.score(X_test, y_test)))
    trainsc = linreg.score(X_train,y_train)
    testsc = linreg.score(X_test, y_test)
    TrainScores2.append(trainsc)
    TestScores2.append(testsc)
# In[]

import matplotlib.pyplot as plt
plt.plot(TrainScores2)
plt.plot(TestScores2)
plt.show()

# In[]
#converting yr_renovated and basement_sqft into 1 and 0
Homeprice['basement_present'] = Homeprice['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not
Homeprice['renovated'] = Homeprice['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) 
# In[]

X = Homeprice[['bedrooms', 'bathrooms','sqft_living','floors','waterfront','view','condition','grade','sqft_above','basement_present','renovated','yr_built','lat','zipcode','long']]
y = Homeprice['price']

# In[]

TrainScores3 = []
TestScores3 = []

for i in range(2,4):
    poly = PolynomialFeatures(degree=i)
    X_F1_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    print('Degree = {:.2f}\nR-squared score (training): {:.3f}'
         .format(i,linreg.score(X_train, y_train)))
    print('(Degree = {:.2f}\nR-squared score (test): {:.3f}\n'
         .format(i,linreg.score(X_test, y_test)))
    trainsc = linreg.score(X_train,y_train)
    testsc = linreg.score(X_test, y_test)
    TrainScores3.append(trainsc)
    TestScores3.append(testsc)

# In[]

import matplotlib.pyplot as plt
plt.plot(TrainScores3)
plt.plot(TestScores3)
plt.show()
