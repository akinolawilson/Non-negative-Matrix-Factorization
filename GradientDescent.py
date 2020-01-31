import numpy as np
import matplotlib.pyplot as plt

def GradientDescent(W,H,V,stepsize):
    WH=np.matmul(W,H)
    WHV=np.subtract(WH,V)
    TransW=np.transpose(W)
    TransH=np.transpose(H)
    WGrad=np.matmul(WHV,TransH)
    HGrad=np.matmul(TransW,WHV)
    Wk1=np.substract(W,stepsize*WGrad)
    Hk1=np.substract(H,stepsize*HGrad)
    n=len(WH)
    for i in range(0,n):
        for j in range (0,n):
            if Wk1<0:
                W[i][j]=0
            else:
                W=Wk1
            if Hk1<0:
                H[i][j]=0
            else:
                H=Hk1
    return(W,H)

def Kullback_Leibler_Divergence(X,V):
    Sum=0.0
    n=len(V)
    for i in range(0,n):
        for j in range(0,n):
            Sum=Sum+X[i][j]*math.log((X[i][j]/V[i][j]),(base))-X[i][j]+V[i][j]
    return (Sum)

V = [[1,2,3],
     [4,5,6],
     [7,8,9]]

W = [[1,1,1],[1,1,1]] 

H = [[1,1],[1,1],[1,1]]

Tol = 1
i=0
stepsize = 0.005
while (Tol > 10e-5):
    X=np.matmul(W,H)
    Tol = Kullback_Leibler_Divergence(X,V)
    Tol_matrix[i]=Tol
    i=i+1
    (W,H)=GradientDescent(W,H,V,stepsize)
    
for x in range(0,i):
    X[x]=x
    
plt.plot(X,Tol_matrix)
plt.xlabel('Iteration')
plt.ylabel('Divergence (Cost)')
    
    
    
