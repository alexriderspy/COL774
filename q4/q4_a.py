import sys
import numpy as np
import pandas as pd

dataX = pd.read_fwf(str(sys.argv[1]) + "/X.dat",header=None)
dataY = pd.read_fwf(str(sys.argv[1]) + "/Y.dat",header=None)

testdataX = pd.read_fwf(str(sys.argv[2]) + "/X.dat", header = None)

arrX = dataX.to_numpy(dtype=float)
arrY = dataY.to_numpy()

def str_to_val(x):
    if x == 'Alaska':
        return 0
    else:
        return 1

vec_Y = np.vectorize(str_to_val)
arrY = vec_Y(arrY)

meanX = np.mean(arrX,axis=0)
stdX = np.std(arrX,axis=0)
arrX = (arrX - meanX)/stdX

m = arrX.shape[0]
n = arrX.shape[1]

arrX.reshape((m,2))
arrY.reshape((m,1))

print(arrX)
print(arrY)

#parameters of the 2 gaussians 
# mu0 for alaska since 0 is alaska

phi = np.sum(arrY)/m
mu1 = np.sum(arrX * arrY,axis=0).reshape((2,1))/np.sum(arrY)
print(mu1)

mu0 = np.sum(arrX * (1-arrY),axis=0).reshape((2,1))/np.sum(1-arrY)
print(mu0)

x = np.array([1,2,3,4]).reshape((2,2))
y= np.array([10,11]).reshape((2,1))
print(x-y)
sigma = (1/m) * np.matmul((arrY*((arrX.T - mu1).reshape((m,2)))+ (1-arrY)*((arrX.T - mu0).reshape((m,2)))).T,(arrY*((arrX.T - mu1).reshape((m,2)))+ (1-arrY)*((arrX.T - mu0).reshape((m,2))))).reshape((2,2))
print(sigma.shape)
print(sigma)
