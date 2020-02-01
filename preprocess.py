import numpy as np
import numpy.linalg as lin
from scipy.optimize import minimize 
import time

class preprocessing:
 


    eps = 0
    precision = 6
    constrainRho = False
    targetSpectralDensity = 0.5
    n = 0
    m = 0
    flattenedX = []
    
    @classmethod
    def setEpsilon(cls, eps):
        cls.eps = eps
    @classmethod
    def setPrecision(cls, precision):
        cls.precision = precision
        
    @classmethod
    def setRho(cls, rho=False):
        '''
        Argument is Boolean
        '''
        if rho == True:
            cls.constrainRho = True
        else:
            cls.constrainRho = False
            
    ########################################
    def __init__(self,
                 eps = 0,
                 precision = 6,
                 constrain_rho = False):
        return 
        
#        self.X = X
#        self.m = X.shape[0]
#        self.n = X.shape[1]
#        self.flattenedX = X.flatten()
        

######################################## just gets dimensions of data set 
            
    @staticmethod
    def Xtest(m,n):
        '''
        Fucntion X returns non-negative, column stochastic m by n matrix
        
        FOR TESTING PURPOSES
        '''
        Xtest = np.random.randint(1,100, m * n)
        Xtest = np.reshape(Xtest,(m,n))
        Xtest = Xtest.astype(float)
        for i in range(np.shape(Xtest)[1]):
            l1Norm = sum(Xtest[:,i])
            Xtest[:,i] = Xtest[:,i] / l1Norm

        return Xtest

    @staticmethod
    def spectralDensity(B):
        return np.abs(lin.eigvals(B)).max()
    
    @staticmethod
    def zeroTruncatedPoisson(loc=1, size=1):
        x = np.random.poisson(loc, size)
        mask = x == 0
        while mask.sum() > 0:
            
            x[mask] = np.random.poisson(loc, mask.sum())
            mask = x == 0
            return x.squeeze()[()]
    
    @staticmethod
    def initialiseB(n):
        ''' Randomly initialize B.
        
        B is an nxn non-negative matrix, with zero diagonal and highest eigenvalue < 1    
        '''
        B = preprocessing.zeroTruncatedPoisson(size=(n,n))
        np.fill_diagonal(B, 0)
        B = B * preprocessing.targetSpectralDensity / preprocessing.spectralDensity(B)

        return B

####################################################################
    @staticmethod
    def diagFlatIndexElementsX(n):
        return np.arange(0, n*n, n + 1)    
     # Tracking index of off-diagonal terms. Index corresponds to index once flattened
    @staticmethod
    def offDiagFlatIndexElementsX(n):
        IndicesX = np.arange(n*n)
        return np.delete(IndicesX, preprocessing.diagFlatIndexElementsX(n) )
    # Converting off diagonal terms into flat string using their corresponding index number
####################################################################



#########################################################################  
    @staticmethod      
    def offDiagElementsToFlat(params,n):
        return np.delete(preprocessing.flatToOffDiag(params),
                         preprocessing.diagFlatIndexElementsX(n) ) # deleting the elements with indices of diag elements
 
    @staticmethod
    def flatToOffDiag(params):
    # argument of function is flattened string of elements that are the 
    # off diagonal elements of the matrix to be pre-processed 
        n = int( np.ceil( np.sqrt( len(params) ) ) )
        Y = np.zeros((n, n), dtype=preprocessing.flattenedX.dtype)
        
        Y.flat[ preprocessing.offDiagFlatIndexElementsX(n) ] = params  
        return Y # , n by n matrix with off diag elements equal to dataset matrix of


#############################################################################
    @staticmethod
    def objective(params, X):
        # p; parameters i.e. the elements of the matrix to be pre-processed.
                
        B = preprocessing.flatToOffDiag(params)
        return lin.norm(X - X @ B)
    
    @staticmethod
    def noiseyOrSparseConstraint(params, X):
    # Replaxing contraint to deal with noisy data, notice 
        B = preprocessing.flatToOffDiag(params)
        
        return (X + preprocessing.eps * lin.norm(X, np.inf) - X @ B).flatten()
    
    @staticmethod
    def spectralRadiusRhoConstraint(params):
        
        B = preprocessing.flatToOffDiag(params)
        rhoB = preprocessing.spectralDensity(B)
        return 1 - rhoB
           
####################################################################        
    @staticmethod
    def preprocess(X): #, X, eps=eps, precision=precision, constrain_rho=constrain_rho):
        '''
        preprocess will attempt to find inverse postive, non-negative matrix B, which
        attempts to maximise the number of 0 elements in M, while still preserving the 
        columns and their relation to one another.
        
        Precision is number of the decimal spaces to be rounded to.
        
        Epsilon is hyperparameter for relaxing constraint of objective function. The 
        parameter is there to aid with dealing with noisy data. 
        
        Function returns: B matrix for scaling, Mprime scaled down M, and Mprime_norm, column 
        stochastic scaled down matrix
        '''
        preprocessing.m = X.shape[0]
        preprocessing.n = X.shape[1]
        preprocessing.flattenedX = X.flatten()
        
        
        start = time.time()
        B = preprocessing.initialiseB(preprocessing.n) # random choice for B
        params = B.flatten() # flatten elements for optimisation
        
        constraints = [{'type': 'ineq',
                        'fun': preprocessing.noiseyOrSparseConstraint,
                        'args':(X,)}] # constraint for epsilon ,notice need arguements of confunc
        
        if preprocessing.constrainRho == True:
            constraints.append({'type': 'ineq',
                                'fun': preprocessing.spectralRadiusRhoConstraint}) # spectrual radius constraint
        
        bounds = [[0, None]] * len(params) # bounding the elements of B, which are contained in p 
        
    
        result = minimize(preprocessing.objective,
                          params,
                          args=(X,),
                          method='SLSQP',
                          constraints=constraints,
                          bounds=bounds)
        
        B = preprocessing.flatToOffDiag(result.x)

        I = np.identity(B.shape[0])
        Xprime = X @ (I - B)
        normFactor = lin.norm(X) / lin.norm(Xprime)
        
        XprimeNorm = Xprime * normFactor
        
        # Avoid tiny values in output matrices
        for x in (B, Xprime, XprimeNorm):
            
            x.round(preprocessing.precision, out=x)
            
            
        end = time.time()
        
        print('General information: \n''Optimisation complete. \n' 
              'Matrix information: dimensions: ({},{}) \n'
              'Optimisation completion time: {:.2f} seconds \n'
              'The first returned matrix corresponds to the generalised'
              ' permuatation matrix, the second the '
              ' processed matrix and the last, is the normalised'
              ' processed matrix'.format((preprocessing.m),
                                            (preprocessing.n),
                                            (end - start)))
   
        
        return B, Xprime, XprimeNorm

#%%
