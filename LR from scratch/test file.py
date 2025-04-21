# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 21:00:42 2025

@author: dheena
"""

from LinearRegression import LinearRegression

import pandas as pd

df = pd.read_csv("Student_Performance.csv")

df["Extracurricular Activities"]=df["Extracurricular Activities"].replace("Yes",1)
df["Extracurricular Activities"]=df["Extracurricular Activities"].replace("No",0)

model=LinearRegression()
print(df.shape)

X=df.drop(["Performance Index"],axis=1)
y=df["Performance Index"]

print(X.shape)
model.fit(X, y)

print(model.predict(X.iloc[0]))

print("score:",model.score(X,y))
      