import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import scipy.stats as stats
import numpy.linalg as lin
#from scipy.optimize import minimize
#from scipy.optimize import Bounds
#from sklearn.decomposition import NMF
import time
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1), copy=True)
#%%
Xtrain = np.load("Z:\\SML\project\Xtrain.npy")
#%%
Xtrain1 = np.load("Z:\\SML\project\Xtrain1.npy")
Xtrain2 = np.load("Z:\\SML\project\Xtrain2.npy")
#%%
tol = 0.3

def XWHapprox(n,m,r):
    np.random.seed(10)
    Mout = np.random.randint(1,100,n*m)
    Mout = np.reshape(Mout, (n,m)).astype(float)
    Wval = np.random.randint(1, 100, n*r)
    Wout = np.reshape(Wval, (100, r)).astype(float)
    Hval = np.random.randint(1, 100, r*m)
    Hout = np.reshape(Hval, (r, 50)).astype(float)
    return(Mout, Wout, Hout)

def check_pos(X):
    X[X < 0] = 0
    X[X == np.inf] = 0
    X[X == -np.inf] = 0
    X_nans = np.isnan(X)
    X[X_nans] = 0
    #dim = X.shape
    #n = dim[0]
    #m = dim[1]
    #for i in range(n):
    #    for j in range(m):
    #        if X[i,j] < 0:
    #           X[i,j] = 0
    #        if X[i,j] == np.inf:
    #            X[i,j] = 0
    #        if X[i,j] == 
    return X

def FnormWH(W):
    Fout = np.sqrt(np.sum(np.power(np.absolute(W),2)))
    return Fout

def FnormXWH(X, W, H, beta=0.1):
    A = X - W@H
    Fout = np.sqrt(np.sum(np.power(np.absolute(A),2))) + beta*(FnormWH(W) + FnormWH(H))
    return Fout

def euclidcol(X):
    colsum = np.sqrt((np.sum(X, axis=0))**2)
    n = X.shape[1]
    for i in range(n):
            X[:,i] = X[:,i]/colsum[i]
    return X

#%%
def initialiseWH(X,r):
    np.random.seed(10)
    Xlen = X.shape[0]
    rmax = X.shape[1]
    Maxval, Minval = [], [] 
    Wval, Hval = [], []
    if r > rmax:
        print("Cannot factorise W and H into larger matrices than X.")
        return (Wval, Hval)
    else:
        for i in range(r):
            Maxval = np.append(Maxval, max(Xtrain1[:,i]))
            Minval = np.append(Minval, min(Xtrain1[:,i]))
        Maxval = np.reshape(Maxval, (1,len(Maxval)))
        Minval = np.reshape(Minval, (1,len(Minval)))
        MinMax = np.append(Maxval, Minval, axis=0)
        for j in range(r):
            Wval = np.append(Wval, np.random.randint(MinMax[1,j], MinMax[0,j], Xlen*1))
            Hval = np.append(Hval, np.random.randint(MinMax[1,j], MinMax[0,j], 1*rmax))
        Wout = np.reshape(Wval, (Xlen,r)).astype(float)
        Hout = np.reshape(Hval, (r,rmax)).astype(float)
        return (Wout, Hout)
#%%
W, H = initialiseWH(Xtrain1, 19)
#%%
def ALS(X,W,H,tolerance,Imax=1000, beta=0.1):
    Wup = W
    Hup = H
    check_pos(Wup)
    check_pos(Hup)
    I = 0
    Fcheck = FnormXWH(X,Wup,Hup,beta)
    Fvec= []
    Fvec = np.append(Fvec, Fcheck)
    
    while Fcheck > tolerance and I < Imax:
        Wup = euclidcol(Wup)
        Winv = check_pos(lin.pinv(Wup.transpose() @ Wup))
        Hup = Winv @ (Wup.transpose() @ X)
        check_pos(Hup)
        
        Hup = euclidcol(Hup)
        Hinv = check_pos(lin.pinv())
        Wup = (W )
        
        
        
        
        
        Wup = euclidcol(Wup)
        Hup = (check_pos(lin.pinv(Wup.transpose())) @ Wup) @ (Wup.transpose() @ X)
        check_pos(Hup)
        
        Hup = euclidcol(Hup)
        Wup = (X @ Hup.transpose()) @ (check_pos(lin.pinv(Hup @ Hup.transpose())))
        check_pos(Wup)
        
        Fcheck = FnormXWH(X,Wup,Hup,beta)
        Fvec=np.append(Fvec, Fcheck)
        I += 1
    
    minval = min(Fvec)
    minvalindex = np.argmin(Fvec)
    plt.rcParams["figure.figsize"] = 15, 7
    plt.rcParams["font.size"] = 18
    plt.plot(Fvec, "b-", label="ALS")
    plt.title("Plot of Frobenius Norm Values vs Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frobenius Norm Values")
    plt.hlines(minval, xmin=0, xmax=I, color="black", linestyles="dashed", label="Minimum Value", alpha=0.5)
    plt.legend(loc=0, fontsize=12)

    if I == Imax:
        print("Max number of iterations performed. Convergence hasn't been found.")
        print("Local minimum value of {} was found at iteration {}." .format(minval, minvalindex))
        return(I, Wup, Hup, Fvec)
    else:
        print("{} iterations performed. Convergence has been found." .format(I))
        return(I, Wup, Hup, Fvec)

#%%
Iterations, Wout, Hout, Fnorm_result = ALS(Xtrain1, W, H, tolerance = tol, Imax=100, beta=0)

#%%
test = euclidcol(H)
test1 = lin.pinv(test)