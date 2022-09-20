# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:24:45 2022

@author: Kinjal Pandya
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os

# reading data
data = pd.read_csv("Automobile_data.csv")

#getting top 5 column
print(data.head())

#getting bottom 5 column
print(data.tail())

#checkingshape
print(data.shape)

###getting statistics
print(data.describe())

#getting column names
print(data.columns)

#getting unique values
print(data.nunique())

#cleaning the data

#checking null values
print(data.isnull().sum())

#checking datatypes
print(data.info())

#Checking for wrong entries like symbols -,?,#,*,etc.
for col in data.columns:
    print('{}:{}'.format(col, data[col].unique()))
    
###replacing ? into np.nan form.
for col in data.columns:
    data[col].replace({'?':np.nan},inplace=True)
print(data.head())

#checking null values
print(data.isnull().sum())

###see the amount of data that is missing from the attribute
sns.heatmap(data.isnull(),cbar=False,cmap='viridis')

###Replacing the missing values
num_col = ['normalized-losses', 'bore',  'stroke', 'horsepower', 'peak-rpm','price']
for col in num_col:
    data[col]=pd.to_numeric(data[col])
    data[col].fillna(data[col].mean(), inplace=True)
print(data.head())

###correlation between different variables
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),cbar=True,annot=True,cmap='Blues')

###How does the horsepower affect the price?
plt.figure(figsize=(10,10))
plt.scatter(x='horsepower',y='price',data=data)
plt.xlabel('Horsepower')
plt.ylabel('Price')

###univariate analysis of horsepower.
sns.histplot(data.horsepower,bins=10)

### What is the relation between engine_size and price?
plt.figure(figsize=(10,10))
plt.scatter(x='engine_size',y='price',data=data)
plt.xlabel('Engine size')
plt.ylabel('Price')

### How does highway_mpg affects price?
plt.figure(figsize=(10,10))
plt.scatter(x='highway_mpg',y='price',data=data)
plt.xlabel('Higway mpg')

###Relation between no. of doors and price
sns.boxplot(x='price',y='num_of_doors',data=data)