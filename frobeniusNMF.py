import numpy as np
import numpy.linalg as lin
import time as t

class FrobeniusNMF:
    
    import numpy as np
    import numpy.linalg as lin
    import time as t    
    
    epsilon = 1e-3
    beta = 0.1
    
    m = 0
    n = 0
    
    iterations = 500
    repetitions = 500

    
    
    def __init__(self, beta = 0.1, epsilon = 1e-3, iterations = 500, repetitions =500):
        
        FrobeniusNMF.beta = beta
        FrobeniusNMF.epsilon = epsilon
        FrobeniusNMF.iterations = iterations
        FrobeniusNMF.repetitions = repetitions

    @classmethod
    def setEpsilon(cls, eps):
        cls.epsilon = eps
    @classmethod
    def setAlpha(cls, alpha):
        cls.alpha = alpha
    @classmethod
    def setBeta(cls, beta):
        cls.beta = beta
    @classmethod    
    def setIterations(cls, its):
        cls.iterations = its        
    @classmethod
    def setRepetitions(cls, reps):
        cls.repetitions = reps 
        
    @staticmethod
    def zeroTruncatedPoisson(loc=1, size=1):
        '''
        Draws random values from zero truncated poisson distribution for initialising 
        purposes. Will return array of specified size containing instances from ZTP distribution
        
        For initialising purposes
        '''
        x = np.random.poisson(loc, size)
        mask = x == 0
        while mask.sum() > 0:
            x[mask] = np.random.poisson(loc, mask.sum())
            mask = x == 0
        return x.squeeze()[()]
    
    
    @staticmethod
    def initialiseWandH(m, n, r):
        '''
        Function to initialise two matrix: Wo - m by r - and Ho - r by n. Where r is the
        rank of the chosen decomposition. Wo and Ho are both non-negative and column stochastic
        matrices.
        
        For testing purposes.
        '''
        Wo = FrobeniusNMF.zeroTruncatedPoisson(size=(m,r))
        Ho = FrobeniusNMF.zeroTruncatedPoisson(size=(r,n))
        Wo = np.reshape(Wo,(m,r)).astype(float)
        Ho = np.reshape(Ho,(r,n)).astype(float)
        
        for i in range(r):
            l1NormWo = sum(Wo[:,i])
            Wo[:,i] = Wo[:,i] / l1NormWo   
        for j in range(n):
            l1NormHo = sum(Ho[:,j])
            Ho[:,j] = Ho[:,j] / l1NormHo
        
        
        return Wo, Ho
    
    @staticmethod
    def checkPos(X):
        X[X < 0] = 0
        X[X == np.inf] = 0
        X[X == -np.inf] = 0
        XNans = np.isnan(X)
        X[XNans] = 0
        return X

    @staticmethod
    def Fnorm(X):
        Fout = np.sqrt(np.sum(np.power(np.absolute(X),2)))
        return Fout
    
    @staticmethod
    def frobeniusObjectiveFunction(X, W, H):
        A = X - W@H
        Fout = np.sqrt( np.sum(np.power(np.absolute(A),2)) ) + FrobeniusNMF.beta*(FrobeniusNMF.Fnorm(W) + FrobeniusNMF.Fnorm(H))
        return Fout
    
    @staticmethod
    def colNormalisation(X):
        colsum = np.sqrt((np.sum(X, axis=0))**2)
        n = X.shape[1]
        for i in range(n):
                X[:,i] = X[:,i]/colsum[i]
        return X   
    
    @staticmethod
    def NNALSOpt(X,
                 r,
                 beta = 0.1,
                 epsilon = 1e-3,
                 iterations = 500,
                 repetitions =500):
        
        FrobeniusNMF.beta = beta
        FrobeniusNMF.epsilon = epsilon
        FrobeniusNMF.iterations = iterations
        FrobeniusNMF.repetitions = repetitions        
        
        
        start = t.time()
        
        Wup, Hup = FrobeniusNMF.initialiseWandH( X.shape[0], X.shape[1], r)
        cost = FrobeniusNMF.frobeniusObjectiveFunction(X,Wup,Hup)

        ite = 0
        rep = 0 
     
        while rep <= FrobeniusNMF.repetitions:

            Wup, Hup = FrobeniusNMF.initialiseWandH( X.shape[0], X.shape[1], r)        
            ite = 0
                
            while cost > FrobeniusNMF.epsilon and ite <= FrobeniusNMF.iterations:
                
                WupKminus1 = np.copy(Wup)
                HupKminus1 = np.copy(Hup)
                
                Winv = FrobeniusNMF.checkPos(lin.pinv(Wup))
                # updating H
                Hup = FrobeniusNMF.checkPos( Winv @ X )
                
                Hinv = FrobeniusNMF.checkPos(lin.pinv(Hup))
                
                # updating W
                Wup = FrobeniusNMF.checkPos(X @ Hinv)
                
                # evaluating cost
                costAfter = FrobeniusNMF.frobeniusObjectiveFunction(X,Wup,Hup)
                costBefore = FrobeniusNMF.frobeniusObjectiveFunction(X,WupKminus1,HupKminus1)
                
                cost = np.abs(costBefore - costAfter)
                
                ite += 1
                
                if ite == FrobeniusNMF.iterations: # if run through iterations, and no sol, repeat
                    rep += 1
                
                
                if cost <= FrobeniusNMF.epsilon:
                    break 
                
            if cost <= FrobeniusNMF.epsilon:
                
                endcomplete = t.time()
                Wprime, Hprime = Wup, Hup
                print('Optimisation complete using Frobenius objective and alternating least squares NMF. \n'
                  ' Proccessing time: {:.3f} seconds. \n'
                  ' Error: {:.8f} \n'
                  ' Parameter information:- \n'
                  '     -rank: {} \n'
                  '     -beta: {} \n'
                  ' Number of re-initialisations required: {}.'
                  ''.format((endcomplete-start), 
                            np.absolute(cost),
                            (r),
                            (FrobeniusNMF.beta),
                            (rep)))
                break
            
            if rep == FrobeniusNMF.repetitions:
                
                endbad = t.time()
                Wprime, Hprime = 0,0
                print('Failed to optimise with achieved error: {:.3f}. \n'
                      ' Try increasing the error tolerance epsilon through the class method \n'
                      ' or increase the repitions of instantiation or iterations \n'
                      ' The parameters current values are: \n'
                      ' Epsilon: {} \n'
                      ' Repitions: {} \n'
                      ' Iterations: {} \n'
                      ' Time wasted: {:.3f} seconds'.format((np.absolute(cost)),
                                                             (FrobeniusNMF.epsilon),
                                                             (FrobeniusNMF.repetitions),
                                                             (FrobeniusNMF.iterations),
                                                             (endbad-start)))
                break
            
            
        return Wprime, Hprime, cost 

#%%