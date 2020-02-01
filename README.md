# Non-negative Matrix Factorization

## Overview 
________________________________________________________________________________________________________________________________________
This repository contains classes for the dimension reduction technique of non-negative matrix factorization alongside a data pre-processing technique. Three different cost functions have been implemented paired with three different optimisation methods. These are:<br>
* Frobenius norm and non-negative alternating least squares 

* Kullback-Leibler divergence and gradient descent 

* Localised divergence and multiplicative updates

### Prerequisites
________________________________________________________________________________________________________________________________________
To make use of the classes, a user would need the following packages:-
* Numpy 
* Scipy

### Examples of use
_______________________________________________________________________________________________________________________________________
#### preprocessing
```
pre = preprocessing
# consider design matrix m by n; X, where there are n instances with m features. 
transform, Xtransformed, XtransformedNormalsied = pre.preprocess(X)
# Where 'transform' is the n by n matrix describing the undergone transformation of the design matrix, 
# 'Xtransformed' is the end result design matrix and 'XtransformedNormalsied' is it normalised across each instance.
```
#### KLNMF
```
Give examples
```
#### frobeniusNMF

```
Give examples
```
#### localNMF
```
Give examples
```
### Further Development and Reading
________________________________________________________________________________________________________________________________________

The next task at hand is decoupling the optimisation methods from the objective functions, such that they can be used independently. A formal review of the models and an example application can be found at my <a href="https://www.researchgate.net/publication/338197703_Non-negative_Matrix_Factorization">ResearchGate</a> account. 
