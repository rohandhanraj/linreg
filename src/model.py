import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import logging

class LinearRegressor:
    def __init__(self, learning_rate= 0.1, n_jobs=None, n_iter=10000):
        self.eta = learning_rate
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        
    def fit(self, X, y):
        
        self.scaler = StandardScaler()
        
        r, c = X.shape
        self.X = self.scaler.fit_transform(X)
        self.y = np.array(y).flatten().reshape(r, 1)
        
        X = np.c_[np.ones((r, 1)), self.X]
        c += 1
        self.w = np.random.randn(c, 1) * 1e-4
        logging.info(f'Initializing...\nIntercept: {self.w[0]},\nCoeffficient: {self.w[1:]}')
        
        for i in tqdm(range(self.n_iter), total=self.n_iter, desc='Training the model', colour = 'green'):
            logging.info('---'*10)
            logging.info(f'# Iteration: {i+1}')
            logging.info('---'*10)
                        
            y_hat = X.dot(self.w) # prediction vector
            
            logging.info(f'RMSE: {self.rmse(self.y, y_hat)}\n')
            
            grads = (2/r) * X.T.dot(y_hat - self.y)
            logging.info(f'dW:\n{grads}\n')

            steps = self.eta * grads
            logging.info(f'Steps:\n{steps}\n')

                        
            self.w = self.w - steps
            self.intercept = self.w[0]
            self.coef = self.w[1:]
            logging.info(f'\nUpdated Intercept and Coefficients after iteration: {i+1}/{self.n_iter}:\nIntercept: {self.intercept},\nCoef:\n{self.coef}')
            logging.info('#####'*10)
                       
            if np.all(np.abs(steps)) < 0.00001:
                break
            else:
                continue
        
        return self
        
            
    def predict(self, X):
        self.scaler.transform(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = X.dot(self.w)
        return y_pred
    
    def rmse(self, y, y_):
        y = np.array(y).flatten().reshape(len(y), 1)
        y_ = np.array(y_).flatten().reshape(len(y_), 1)
        rmse = np.sqrt(np.mean((y - y_)**2))
        return rmse
    
    def score(self, x, y):
        y_ = self.predict(x)
        y = np.array(y).flatten().reshape(len(y), 1)
        ym = np.mean(y)
        rss = sum((y - y_)**2)
        tss = sum((ym - y_)**2)
        r2 = 1 - (rss / tss)
        return r2
    
    def adj_r2(self, x, y):
        r2 = self.score(x,y)
        n = x.shape[0]
        p = x.shape[1]
        adj_r2 = 1 - ((1 - r2) * (n-1) / (n-p-1))
        return adj_r2