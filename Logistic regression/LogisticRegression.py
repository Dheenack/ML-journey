# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:02:30 2025

@author: dheena 

Code for Logistic Regression
"""
import numpy as np

def sigmoid(x):
    y=1/(1+np.exp(-x))
    return y

class LogisticRegression():
    
    def __init__(self,lr=0.0001,n_iters=1000):
        self.lr=lr 
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
        
    def fit(self,X,y):
        
        n_rows,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weights) + self.bias
            y_pred = sigmoid(y_pred)
            dw = (1/n_rows)*np.dot(X.T,y_pred-y)
            db = (1/n_rows)*np.sum(y_pred-y)
            
            self.weights-=self.lr*dw
            self.bias-=self.lr*db
            
    def predict(self,X):
        y_pred = np.dot(X,self.weights) + self.bias
        y = sigmoid(y_pred)
        y=np.where(y>0.5,1,0)
        return y
    