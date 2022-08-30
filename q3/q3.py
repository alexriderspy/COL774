import sys
import numpy as np
import pandas as pd
import math

dataX = pd.read_csv(str(sys.argv[1]) + "/X.csv",header=None)
dataY = pd.read_csv(str(sys.argv[1]) + "/Y.csv",header=None)

testdataX = pd.read_csv(str(sys.argv[2]) + "/X.csv", header = None)

arrX = dataX.to_numpy()
arrY = dataY.to_numpy()

meanX = np.mean(arrX)
stdX = np.std(arrX)
arrX = (arrX - meanX)/stdX

m = arrX.shape[0]
n = arrX.shape[1]
z = np.ones((m,1))

arrX = np.append(arrX,z,axis=1)
arrX[:, [1,2]] = arrX[:,[2,1]]
arrX[:, [0,1]] = arrX[:,[1,0]]

arrX.reshape((m,3))
arrY.reshape((m,1))

theta = np.zeros(3).reshape((3,1))

def g(x):
    return 1/(1+math.exp(-x))

def gradient(theta):
    return np.sum(np.multiply(arrX,arrY - g(np.matmul(arrX,theta))).T,axis=1).reshape((3,1))

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

x = np.random.randn(100, 100, 100)
hessian(x)