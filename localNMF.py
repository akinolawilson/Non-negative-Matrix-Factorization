import numpy as np
import numpy.linalg as lin
import time as t


class localNMF:
    
    epsilon = 1e-3
    alpha = 0.1
    beta = 0.1
    
    m = 0
    n = 0
    r = 0
    
    iterations = 500
    repetitions = 500
    
    
    def __init__(self, alpha = 0.1, beta = 0.1, epsilon = 1e-3, iterations = 500, repetitions =500):
        localNMF.alpha = alpha
        localNMF.beta = beta
        localNMF.epsilon = epsilon
        localNMF.iterations = iterations
        localNMF.repetitions = repetitions 
    
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
        Wo = localNMF.zeroTruncatedPoisson(size=(m,r))
        Ho = localNMF.zeroTruncatedPoisson(size=(r,n))
        Wo = np.reshape(Wo,(m,r)).astype(float)
        Ho = np.reshape(Ho,(r,n)).astype(float)
        
        for i in range(r):
            l1NormWo = sum(Wo[:,i])
            Wo[:,i] = Wo[:,i] / l1NormWo   
        for j in range(n):
            l1NormHo = sum(Ho[:,j])
            Ho[:,j] = Ho[:,j] / l1NormHo
        
        
        return Wo, Ho
    
    # residuals = lin.norm(ianNorm - w @ h) # we seek to minimise this value 

        
    ###############################################################################
    @staticmethod
    def logIssue(a, b):
        '''
        Function attempts to deal with argument of zero in localNMF objective function.
        
        
        '''
        if b == 0 or b <= localNMF.epsilon/20 :
           b = localNMF.epsilon/2   
            
        if a/b == 0 or a/b <= localNMF.epsilon/20 :
           a = a + localNMF.epsilon 
           b = 2*localNMF.epsilon
           
        result = np.log( a / b )
        
        return result
    
        
    ###############################################################################
    # Found zero values in image, need to replace with small values, e-7
    
    @staticmethod
    def localNMFobjective(X, W, H):
        '''
        Local NMF imposed constraints on spatial locality of features. This is done 
        by extending the divergence algorithm to include additional constraints, whose
        importance/impact on the objective function is quantified by alpha and beta, where
        these are postive constants.
        '''

        
        V,U,Z = [],[],[] 
        X = np.where(X==0, 1e-7, X) # zeros will cause issues in logarithm
        localNMF.n = np.shape(X)[1]
        localNMF.m = np.shape(X)[0]
        localNMF.r = np.shape(W)[1]
        
        
        for i in range(localNMF.m):
            for j in range(localNMF.n):
                
                
                
                V.append( X[i,j] * localNMF.logIssue( X[i,j] , (W@H)[i,j]) - X[i,j] + (W@H)[i,j] )    
                
        for k in range(localNMF.r):
            for l in range(localNMF.r):
              U.append( (np.transpose(W)@W)[k,l] )
        for q in range(localNMF.r):
            Z.append( (H@np.transpose(H))[q,q] )    
        
        U, V, Z = (1/len(U))*sum(U), (1/len(V))*sum(V), (1/len(Z))*sum(Z) 
        
        
        return (V + localNMF.alpha*U - localNMF.beta *Z)
    
    ###############################################################################
    
    @staticmethod
    def interationFunction(X, Wo, Ho):
        
        '''
        peforms itertion for optimsation in Multiplicative update rule of Kullback- Leiber 
        divergence formulation of cost function associated to local NMF. 
        
        '''
        
        num = 0
        while  num <= localNMF.iterations:
           
            num += 1
            
            for i in range(localNMF.r):
                 for j in range(localNMF.n):
                     numerator = 0
                     denominator = 0
                     for k in range(localNMF.m):
                         numerator +=( (Wo[k,i] * X[k,j])  / (Wo @ Ho)[k,j] )
                     for l in range(localNMF.m):
                         denominator += Wo[l,i]
                    
                     Ho[i,j] = Ho[i,j] * ( numerator / denominator )
                    
    
                    
            for i in range(localNMF.m):
                for j in range(localNMF.r):
                
                     numerator = 0
                     denominator = 0       
                     for k in range(localNMF.n):
                         numerator += ((Ho[j,k] * X[i,k]) / (Wo @ Ho)[i,k] )
                     for l in range(localNMF.n):
                         denominator += Ho[j,l]
                        
                     Wo[i,j] = Wo[i,j] * ( numerator / denominator )
            
        
        return Wo, Ho
    ###############################################################################
    @staticmethod
    def LNMF(X,
             r,
             alpha=0.1, 
             beta=0.1,
             iterations=500,
             repetitions=500,
             epsilon=0.01):
 
        '''
        This function describes the multiplicative update rule for optimising W and H
        whose product is approximately X, the data matrix. The optimised matrices for W
        and H will be returned along with the error of the approximation. This error
        can be used for optimising hyperparameters such as alpha and beta. Furthermore, the
        preprocessing function also has a hyperparameter, which the error of this function 
        will be useful for.
        '''
        finished = False
        start = t.time()
        localNMF.r = r
        localNMF.alpha = alpha
        localNMF.beta = beta
        localNMF.iterations = iterations 
        localNMF.repetitions = repetitions
        localNMF.epsilon = epsilon
        
        rep = 0
        Wprime = None
        Hprime = None
    
    
        while rep <= repetitions:
            
            rep += 1
            m, n = np.shape(X)[0], np.shape(X)[1]
            Wo, Ho = localNMF.initialiseWandH(m,n,r)
            
            if rep in np.arange(0, rep, int(rep/10) + 1):
                print('computing...')
                
            Woit, Hoit = localNMF.interationFunction(X, Wo, Ho)
            error = localNMF.localNMFobjective(X, Woit, Hoit)
            
            if (epsilon >= np.absolute( error )):
                finished = True 
                break
            
            if rep == repetitions-1 and finished == False:
                endb = t.time()
                Wprime = None
                Hprime = None
                error = error
                print('Failed to optimise with achieved error: {:.3f}. \n'
                      ' Try increasing the error tolerance epsilon through the class method \n'
                      ' or increase the repitions of instantiation or iterations \n'
                      ' The parameters current values are: \n'
                      ' Epsilon: {} \n'
                      ' Repitions: {} \n'
                      ' Iterations: {} \n'
                      ' Time wasted: {:.3f} seconds'.format((np.absolute(error)),
                                                             (localNMF.epsilon),
                                                             (localNMF.repetitions),
                                                             (localNMF.iterations),
                                                             (endb-start)))
            

        if finished == True:
                
            endt = t.time()
            Wprime = np.where(Woit < epsilon*1e-3, 0, Woit) # drop values orders of three
            Hprime = np.where(Hoit < epsilon*1e-3, 0, Hoit) # lower than error tolerence, epsilon
            error = error
            print('Optimisation complete using localised NMF. \n'
                  ' Proccessing time: {:.3f} seconds. \n'
                  ' Error: {} \n'
                  ' Parameter information:- \n'
                  '     -rank: {} \n'
                  '     -alpha: {} \n'
                  '     -beta: {} \n'
                  ' Number of re-initialisations required: {}.'
                  ''.format((endt-start), 
                            np.absolute(error),
                            (localNMF.r),
                            (localNMF.alpha),
                            (localNMF.beta),
                            (rep)))

    
        return Wprime, Hprime, error
    

#%%
