# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 21:20:31 2022

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
df = pd.read_csv("brain_body.csv")
print(df)
# here we divide data frame into 2 parts because fit method take 2 dimension array so we
make one
# for brain two dimension array and second for body weight
# x = independent variable in my vase brain weight
# y = dependent variable in my case body weight
brain_df = df.drop("body_weight",axis="columns")
print(brain_df)
# here we train our model for test
model = linear_model.LinearRegression()
model.fit(brain_df, df.body_weight)
#after we train our model we predict new body weight for given brain weigh
new_body_weight = model.predict([[15]])
print(new_body_weight)
# this is called slope for given formula
print("coefficient is = ", model.coef_)
# this is intercept
print("intercept is = ", model.intercept_)
body_weight_predict = model.predict(brain_df)
plt.xlabel("brain weight") #independent variable
plt.ylabel("body weight") # dependent variable
plt.scatter(df.brain_weight, df.body_weight,color="red",marker="*")
plt.plot(df.brain_weight, body_weight_predict,color="blue")
plt.show()
# formula for prediction y = m * x + b
#y = 4.12493507 * 15 + 17.6401148368452
#print("y is = ", y)