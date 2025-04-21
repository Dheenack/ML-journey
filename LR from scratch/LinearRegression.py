# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 20:36:07 2025

@author: dheena krishna

"""

import numpy as np

class LinearRegression():
    
    def __init__(self,learning_rate=0.0001,n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self,X,y):
        
        n_rows,n_features = X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weights) + self.bias
            dw = (-2/n_rows)*np.dot(X.T,y-y_pred)
            db = (-2/n_rows)*np.sum(y-y_pred)
            print(f"Iteration {_}: Loss = {np.mean((y - y_pred) ** 2)}")
            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db
    
    def predict(self,X):
        
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
            
        