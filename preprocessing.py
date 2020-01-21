import numpy as np
import numpy.linalg as lin
from scipy.optimize import minimize 
import time

def colNormalisation(X):
    for i in range(np.shape(X)[1]):
        X[:,i] = X[:,i]/sum(X[:,i])
        
    return X

def zero_truncated_poisson(loc=1, size=1):
    x = np.random.poisson(loc, size)
    mask = x == 0
    while mask.sum() > 0:
        x[mask] = np.random.poisson(loc, mask.sum())
        mask = x == 0
    return x.squeeze()[()]

def spectral_density(Y):
    return np.abs(lin.eigvals(Y)).max()

def initialise_B(n):
    ''' Randomly initialize B.
    
    B is an nxn non-negative matrix, with zero diagonal and highest eigenvalue < 1    
    '''
    target_spectral_density = 0.5
    B = zero_truncated_poisson(size=(n, n))
    np.fill_diagonal(B, 0)
    B = B * target_spectral_density / spectral_density(B)
    return(B)

def X(m,n): 
    '''
    Fucntion X returns non-negative, column stochastic m by n matrix
    
    FOR TESTING PURPOSES
    '''
    X = np.random.randint(1,100,m*n)
    X = np.reshape(X,(m,n))
    X = X.astype(float)
    for i in range(np.shape(X)[1]):
        l1Norm = sum(X[:,i])
        X[:,i] = X[:,i] / l1Norm
        
        return(X)

 # Tracking index of diagonal terms. Index corresponds to index once flattened   
def diag_flatidx(n):
    return np.arange(0, n*n, n + 1)    
 # Tracking index of off-diagonal terms. Index corresponds to index once flattened   
def offdiag_flatidx(n):
    idx = np.arange(n*n)
    return np.delete(idx, diag_flatidx(n))
# Converting off diagonal terms into flat string using their corresponding index number
def offdiag_to_flat(Y):
    n = Y.shape[0]
    return np.delete(Y, diag_flatidx(n)) # deleting the elements with indices of diag elements
# Converting flat string of off diagonal terms into matrix.  
    
def flat_to_offdiag(x):
    n = int(np.ceil(np.sqrt(len(x))))
    Y = np.zeros((n, n), dtype=x.dtype)
    Y.flat[offdiag_flatidx(n)] = x
    return Y


def objective(p, X):
    B = flat_to_offdiag(p)
    return lin.norm(X - X@B)

def confunc(p, X, eps=0):
# Replaxing contraint to deal with noisy data, notice 
    B = flat_to_offdiag(p)
    return (X + eps * lin.norm(X, np.inf) - X@B).flatten()

def conrhofunc(p):
    B = flat_to_offdiag(p)
    rhoB = spectral_density(B)
    return 1 - rhoB

def preprocess(X, eps=0, precision=6, constrain_rho=False):
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
    B = initialise_B(np.shape(X)[1]) # random choice for B
    p = offdiag_to_flat(B) # flatten elements for optimisation
    constraints = [{'type': 'ineq', 'fun': confunc, 'args': (X, eps)}] # constraint for epsilon ,notice need arguements of confunc
    if constrain_rho:
        constraints.append({'type': 'ineq', 'fun': conrhofunc}) # spectrual radius constraint
    bounds = [[0, None]] * len(p) # bounding the elements of B, which are contained in p 
    result = minimize(objective, p, args=(X,), method='SLSQP', constraints=constraints, bounds=bounds)
    B = flat_to_offdiag(result.x)
    I = np.identity(B.shape[0])
    Xprime = X @ (I - B)
    norm_factor = lin.norm(X) / lin.norm(Xprime)
    Xprime_norm = Xprime * norm_factor
    # Avoid tiny values in output matrices
    for x in (B, Xprime, Xprime_norm):
        x.round(precision, out=x)
    end = time.time()
    print('Processing of generalized permutation matrix of X, with X having'
          ' dimensions {} takes {:.2f} seconds'.format(np.shape(X),end - start))
    return B, Xprime, Xprime_norm


from PIL import Image
def imgSizeReduction(X,scale):
    '''
    Function data matrix rescales the images in each column, keeping the aspect ratio
    and returns a data matrix, containing the rescaled images in the columns. The scale 
    arguement works as follow: scale = 2 -> reduce image size by half. 
    
    ASSUMING IMAGES ARE 200 BY 180!!!
    
    '''
    mNew, n = 1596, np.shape(X)[1]  #int(np.shape(X)[0]/scale**4)
    
    newDataMatrix = np.zeros((mNew,n))
    
    for i in range(np.shape(X)[1]):
        
        Xs = np.reshape(X[:,i], (200,180))
        
        img = Image.fromarray(Xs)
        
        
        img.thumbnail(( np.shape(X)[1]/scale, np.shape(X)[0]/scale ))
        
        
        Xsmall = np.array(img)
        size = np.shape(Xsmall)[0]  * np.shape(Xsmall)[1]
         
        newDataMatrix[:,i] = Xsmall.flatten() # , (size,1))
    
    for i in range(np.shape(newDataMatrix)[1]):
        newDataMatrix[:,i] = newDataMatrix[:,i]/sum(newDataMatrix[:,i])
        
        
    return newDataMatrix
