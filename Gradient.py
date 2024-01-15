import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

datX=np.load('x_train.npy')
datY=np.log(np.load('y_train.npy'))
datX=pd.DataFrame(datX, columns=datX.dtype.names)
f, ax=plt.subplots(4, 4, figsize=(16,16))

#your code goes here
cols = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'condition',
           'grade', 'sqft_above', 'sqft_basement', 'long', 'lat']

X = datX[cols]
ones = np.ones((len(X),1))
X = np.concatenate((ones, X), axis=1)
N = len(X)
m = len(X[0])

def loss(w, X, y):
    #your code goes here
    lossValue = (X @ w - y) ** 2
    return lossValue

def grad(w_k, X, y):
    #your code goes here
    loss = X @ w_k.T - y.T
    lossGradient = 2 / N * X.T @ loss
    print(lossGradient)
    return lossGradient.T


ws = np.ones((11,))
loss_val = loss(ws, X, datY)
print(np.shape(loss_val))
gr = grad(ws,X, datY)
print(np.shape(gr))