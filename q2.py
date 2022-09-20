# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:08:56 2022

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("petrol_consumption.csv")
print(df.to_string())
model = linear_model.LinearRegression()
model.fit(df.drop("Petrol_Consumption",axis="columns"),df.Petrol_Consumption)
print(model.coef_)
print(model.intercept_)
# if petrol tax 13 avg_income 5319 paved_highways 1400 population 0.564
petrol_consumes = model.predict([[13,5089,1200,0.634]])
print("petrol consumes:",petrol_consumes)