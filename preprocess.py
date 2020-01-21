import numpy as np
import numpy.linalg as lin
from scipy.optimize import minimize 
import time

class preprocessing:
 


    eps = 0
    precision = 6
    constrainRho = False
    targetSpectralDensity = 0.5
    
    @classmethod
    def setOptimisationParameterEpsilon(cls, eps):
        cls.eps = eps
    @classmethod
    def setOptimisationParameterPrecision(cls, precision):
        cls.precision = precision
        
    @classmethod
    def setOptimisationParameterRho(cls, rho=False):
        '''
        Argument is Boolean
        '''
        if rho == True:
            cls.constrainRho = True
        else:
            cls.constrainRho = False
            
    ########################################
    def __init__(self,
                 X,
                 eps = 0,
                 precision = 6,
                 constrain_rho = False):
        
        self.X = X
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.flattenedX = X.flatten()
        

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
    
    
    def initialiseB(self):
        ''' Randomly initialize B.
        
        B is an nxn non-negative matrix, with zero diagonal and highest eigenvalue < 1    
        '''
        B = self.zeroTruncatedPoisson(size=(self.n, self.n))
        np.fill_diagonal(B, 0)
        B = B * self.targetSpectralDensity / self.spectralDensity(B)

        return B

####################################################################
    def diagFlatIndexElementsX(self):
        return np.arange(0, self.n*self.n, self.n + 1)    
     # Tracking index of off-diagonal terms. Index corresponds to index once flattened
     
    def offDiagFlatIndexElementsX(self):
        IndicesX = np.arange(self.n*self.n)
        return np.delete(IndicesX, self.diagFlatIndexElementsX() )
    # Converting off diagonal terms into flat string using their corresponding index number
####################################################################



#########################################################################        
    def offDiagElementsToFlat(self, params):
        return np.delete(self.flatToOffDiag(params) , self.diagFlatIndexElementsX() ) # deleting the elements with indices of diag elements
 
    
    def flatToOffDiag(self, params):
    # argument of function is flattened string of elements that are the 
    # off diagonal elements of the matrix to be pre-processed 
        n = int( np.ceil( np.sqrt( len(params) ) ) )
        Y = np.zeros((n, n), dtype=self.flattenedX.dtype)
        
        Y.flat[ self.offDiagFlatIndexElementsX() ] = params  
        return Y # , n by n matrix with off diag elements equal to dataset matrix of


#############################################################################    
    def objective(self, params, X):
        # p; parameters i.e. the elements of the matrix to be pre-processed.
                
        B = self.flatToOffDiag(params)
        return lin.norm(X - X @ B)
    
    
    def noiseyOrSparseConstraint(self, params, X):
    # Replaxing contraint to deal with noisy data, notice 
        B = self.flatToOffDiag(params)
        
        return (X + self.eps * lin.norm(X, np.inf) - self.X @ B).flatten()
    

    def spectralRadiusRhoConstraint(self, params):
        
        B = self.flatToOffDiag(params)
        rhoB = self.spectralDensity(B)
        return 1 - rhoB
           
####################################################################        
        
    def preprocess(self, X): #, X, eps=eps, precision=precision, constrain_rho=constrain_rho):
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
        start = time.time()
        X = self.X
        B = self.initialiseB() # random choice for B
        params = B.flatten() # flatten elements for optimisation
        
        constraints = [{'type': 'ineq',
                        'fun': self.noiseyOrSparseConstraint,
                        'args':(X,)}] # constraint for epsilon ,notice need arguements of confunc
        
        if self.constrainRho == True:
            constraints.append({'type': 'ineq',
                                'fun': self.spectralRadiusRhoConstraint}) # spectrual radius constraint
        
        bounds = [[0, None]] * len(params) # bounding the elements of B, which are contained in p 
        
    
        result = minimize(self.objective,
                          params,
                          args=(X,),
                          method='SLSQP',
                          constraints=constraints,
                          bounds=bounds)
        
        B = self.flatToOffDiag(result.x)

        I = np.identity(B.shape[0])
        Xprime = self.X @ (I - B)
        normFactor = lin.norm(X) / lin.norm(Xprime)
        
        XprimeNorm = Xprime * normFactor
        
        # Avoid tiny values in output matrices
        for x in (B, Xprime, XprimeNorm):
            
            x.round(self.precision, out=x)
            
            
        end = time.time()
        
        optimisationInfo = {'General information':'Optimisation complete.',
                            'Matrix information, dimensions: ': '({},{})'.format(self.m, self.n),
                            'Optimisation completion time: ': '{:.2f} seconds'.format(end - start)}
        print(optimisationInfo)
        
        return B, Xprime, XprimeNorm

#%%