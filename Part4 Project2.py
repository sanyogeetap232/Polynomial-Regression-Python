# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:19:12 2017

@author: Sanyu
"""

#part 4 project 2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# In[]
# read the file

homeprice = pd.read_csv("C:\Anaconda\home_price.csv")
homeprice.describe()

# In[]
# plot the autocorrelation

sn.jointplot(x="sqft_living", y="price", data=homeprice, kind = 'reg', size = 7)
sn.jointplot(x="sqft_lot", y="price", data=homeprice, kind = 'reg', size = 5)
sn.jointplot(x="sqft_above", y="price", data=homeprice, kind = 'reg', size = 5)
sn.jointplot(x="sqft_basement", y="price", data=homeprice, kind = 'reg', size = 5)
sn.jointplot(x="sqft_living15", y="price", data=homeprice, kind = 'reg', size = 5)
sn.jointplot(x="sqft_lot15", y="price", data=homeprice, kind = 'reg', size = 5)
sn.jointplot(x="yr_built", y="price", data=homeprice, kind = 'reg', size = 5)
sn.jointplot(x="yr_renovated", y="price", data=homeprice, kind = 'reg', size = 5)
sn.jointplot(x="lat", y="price", data=homeprice, kind = 'reg', size = 5)
sn.jointplot(x="long", y="price", data=homeprice, kind = 'reg', size = 5)
plt.show()

# we can see sqft_lot, sqft_lot15,long arent a good predictors of price

# In[]
#encoding zipcode
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
homeprice["Zip"] = lb_make.fit_transform(homeprice["zipcode"])
# In[]
#converting categorical variables into 1's and 0's

homeprice['basement_present'] = homeprice['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not
homeprice['renovated'] = homeprice['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated

# In[]

X = homeprice[['bedrooms','bathrooms','sqft_living','floors','waterfront','view','condition','grade','basement_present','renovated','sqft_living15','lat','Zip']]
y = homeprice['price']

# In[]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# In[]
# normalize the data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

# In[]
#knn
for i in range(1,5):    
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train_scaled, y_train)
    print('K: {:.2f}'.format(i))
    print('Accuracy of K-NN classifier on training set: {:.2f}'
         .format(knn.score(X_train_scaled, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
         .format(knn.score(X_test_scaled, y_test)))
    

# In[]
# Validation

train, validate, test = np.split(homeprice.sample(frac=1), [int(.6*len(homeprice)), int(.8*len(homeprice))])

# In[]


X_train = train[['bedrooms','bathrooms','sqft_living','floors','waterfront','view','condition','grade','basement_present','renovated','sqft_living15','lat']]
y_train = train['price']

# In[]


X_test = test[['bedrooms','bathrooms','sqft_living','floors','waterfront','view','condition','grade','basement_present','renovated','sqft_living15','lat']]
y_test = test['price']

# In[]


X_validate = validate[['bedrooms','bathrooms','sqft_living','floors','waterfront','view','condition','grade','basement_present','renovated','sqft_living15','lat']]
y_validate = validate['price']

# In[]
# scale the parameters

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)
X_Validate_scaled = scaler.transform(X_validate)

# In[]

for i in range(1,5):
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn.fit(X_train_scaled, y_train)
    print('K: {:.2f}'.format(i))
    print('Accuracy of K-NN classifier on training set: {:.2f}'
         .format(knn.score(X_train_scaled, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
         .format(knn.score(X_test_scaled, y_test)))
    print('Accuracy of K-NN regression on validation set: {:.2f}'
          .format(knn.score(X_Validate_scaled, y_validate)))
    
# In[]

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
         .format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
         .format(knn.score(X_test_scaled, y_test)))
# In[]
# predict

pred = print(knn.predict(X_test_scaled))
pred
