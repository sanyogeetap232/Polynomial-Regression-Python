# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:46:12 2017

@author: Sanyu
"""

# project2 part3

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# In[]

homeprice = pd.read_csv("C:\Anaconda\home_price.csv")
homeprice.describe()

# In[]
#coding for zipcodes
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
homeprice["Zip"] = lb_make.fit_transform(homeprice["zipcode"])


# In[]
# renovated and basement
homeprice['basement_present'] = homeprice['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not
homeprice['renovated'] = homeprice['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated

# In[]

X = homeprice[['bedrooms', 'bathrooms','sqft_living','floors','waterfront','view','grade','sqft_above','basement_present','renovated','lat','long', 'Zip']]
y = homeprice['price']

# In[]
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
    print('Degree = {:.2f}\nR-squared score (test): {:.3f}\n'
         .format(i,linreg.score(X_test, y_test)))
    trainsc = linreg.score(X_train, y_train)
    testsc = linreg.score(X_test, y_test)
    TrainScores.append(trainsc)
    TestScores.append(testsc)
print(TrainScores)
print(TestScores)
# In[]
# visualization

import matplotlib.pyplot as plt
plt.plot(TrainScores)
plt.plot(TestScores)
#plt.axis([0,1,0,1])
plt.show()

# In[]
#L2 penalty

from sklearn.linear_model import Ridge
print('Ridge regression: effect of alpha regularization parameter\n')
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    for i in range(2,4):
        poly = PolynomialFeatures(degree=i)
        X_F1_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0)    
        linridge = Ridge(alpha = this_alpha).fit(X_train, y_train)
        r2_train = linridge.score(X_train, y_train)
        r2_test = linridge.score(X_test, y_test)
        num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
        print('Degree = {:.2f}'.format(i))
        print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))
        

# In[]
RTrainScores = []
RTestScores = []
for i in range(2,4):
        poly = PolynomialFeatures(degree=i)
        X_F1_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0)    
        linridge = Ridge(alpha = 1000).fit(X_train, y_train)
        r2_train = linridge.score(X_train, y_train)
        r2_test = linridge.score(X_test, y_test)
        RTrainScores.append(r2_train)
        RTestScores.append(r2_test)
print(RTrainScores)
print(RTestScores)   

# In[]
# visualization
import matplotlib.pyplot as plt
plt.plot(RTrainScores)
plt.plot(RTestScores)
#plt.axis([0,1,0,1])
plt.show()
   
# In[]
# L1 Penalty
from sklearn.linear_model import Lasso
print('Lasso regression: effect of alpha regularization\nparameter on number of features kept in final model\n')
for alpha in [0.5, 1, 5, 10, 20, 50]:
     for i in range(2,4):
        poly = PolynomialFeatures(degree=i)
        X_F1_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0) 
        linlasso = Lasso(alpha, max_iter = 10000).fit(X_train, y_train)
        r2_train = linlasso.score(X_train, y_train)
        r2_test = linlasso.score(X_test, y_test)
        print('Degree = {:.2f}'.format(i))
        print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))

# In[]
#cross validation for ridge 
from sklearn.model_selection import cross_val_score
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    for i in range(2,4):
        poly = PolynomialFeatures(degree=i)
        X_F1_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0)
        linridge = Ridge(alpha = this_alpha).fit(X_train, y_train)
        cv_scores = cross_val_score(linridge, X_F1_poly, y)
        print('Degree = {:.2f}'.format(i))
        print('Alpha = {:.2f}'.format(this_alpha))
        print('Cross-validation scores (3-fold):', cv_scores)
        print('Mean cross-validation score (3-fold): {:.3f}'
             .format(np.mean(cv_scores)))

# In[]
#cross validation for lasso
for alpha in [0.5, 1, 5, 10, 20, 50]:
     for i in range(1,4):
        poly = PolynomialFeatures(degree=i)
        X_F1_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0) 
        linlasso = Lasso(alpha, max_iter = 10000).fit(X_train, y_train)
        cv_scores = cross_val_score(linridge, X_F1_poly, y)
        print('Degree = {:.2f}'.format(i))
        print('Alpha = {:.2f}'.format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))
        print('Cross-validation scores (3-fold):', cv_scores)
        print('Mean cross-validation score (3-fold): {:.3f}'
             .format(np.mean(cv_scores)))

# In[]
# Coefficients for lasso

poly = PolynomialFeatures(degree=3)
X_F1_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y,
                                                       random_state = 0) 
linlasso = Lasso(alpha, max_iter = 10000).fit(X_train, y_train)
for e in sorted (list(zip(list(X_train), linlasso.coef_)),
                key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))




