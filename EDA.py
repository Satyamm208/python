# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 22:54:24 2022

@author: Admin
"""
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv("petrol_consumption.csv")

##Understanding the data
print(data.head())
print(data.tail())

print(data.shape)
print(data.dtypes)
print(data.describe())
print(data.columns)
print(data.nunique())
print(data['Petrol_tax'].unique())

##cleaning the data
print("+++++++++++++++++null value+++++++++++++++++++=")
print(data.isnull().sum())

data2 = data.drop(['Population_Driver_licence(%)'],axis=1)
print(data2.head())            

#relationship analysis

corelation = data.corr()
sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)

sns.pairplot(data)

sns.relplot(x='Petrol_tax',y='Petrol_Consumption',hue='Average_income',data=data)

sns.displot(data['Average_income'])
sns.catplot(x='Average_income' , kind='box', data=data)