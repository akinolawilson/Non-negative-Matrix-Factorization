# -*- coding: utf-8 -*-
"""University of Birmingham
Created on Fri Nov 29 13:48:45 2019

@author: pmykaw
"""

def Kullback_Leibler_Divergence(X,V):
    Sum=0.0
    n=len(V)
    for i in range(0,n):
        for j in range(0,n):
            Sum=Sum+X[i][j]*math.log((X[i][j]/V[i][j]),(base))-X[i][j]+V[i][j]
        
    

