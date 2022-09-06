import sys
import numpy as np
import pandas as pd
import math

dataX = pd.read_csv(str(sys.argv[1]) + "/X.csv",header=None)
dataY = pd.read_csv(str(sys.argv[1]) + "/Y.csv",header=None)

testdataX = pd.read_csv(str(sys.argv[2]) + "/X.csv", header = None)

arrX = dataX.to_numpy()
arrY = dataY.to_numpy()

meanX = np.mean(arrX,axis=0)
stdX = np.std(arrX,axis=0)
arrX = (arrX - meanX)/stdX

m = arrX.shape[0]
n = arrX.shape[1]
z = np.ones((m,1))

arrX = np.append(arrX,z,axis=1)
arrX[:, [1,2]] = arrX[:,[2,1]]
arrX[:, [0,1]] = arrX[:,[1,0]]

arrX.reshape((m,3))
arrY.reshape((m,1))

def find_ll(arrX,arrY,theta):
  ll_fn = np.sum(arrY * (np.log(1/(1+np.exp(-np.matmul(arrX,theta))))) + (1-arrY) * (np.log(1-1/(1+np.exp(-np.matmul(arrX,theta))))))
  return ll_fn

def find_cost(learned_Y,arrY):
  cost_fn = (1/(2*m))*np.sum((arrY - learned_Y)**2)
  return cost_fn

def commn(theta):
    return -np.exp(-np.matmul(arrX,theta))/(1+np.exp(-np.matmul(arrX,theta)))**2

def inv_hessian(theta):
    common = commn(theta)
    hessian = np.array([np.sum(common),np.sum(common * arrX[:,1]),np.sum(common * arrX[:,2]),np.sum(common * arrX[:,1]),np.sum(common * arrX[:,1] * arrX[:,1]),np.sum(common * arrX[:,1] * arrX[:,2]),np.sum(common * arrX[:,2]),np.sum(common * arrX[:,1] * arrX[:,2]),np.sum(common * arrX[:,2] * arrX[:,2])]).reshape((3,3))
    
    inv_h = np.linalg.pinv(hessian)
    return inv_h

def grad(theta):
    return np.sum((arrX * (arrY - 1/(1+np.exp(-np.matmul(arrX,theta))))).T,axis=1).reshape((3,1))

def newton():
    theta = np.zeros(3).reshape((3,1))
    theta_delta = 1000
    while theta_delta > 1e-9:
        newtheta = theta - np.matmul(inv_hessian(theta),grad(theta))  
        theta_delta = np.sum((newtheta - theta)**2)
        theta = newtheta
    return theta

theta = newton()
print(theta)

Y_pred = 1/(1+np.exp(-np.matmul(arrX,theta)))

def f(x):
    if x > 0.5:
        return 1
    else:
        return 0

f_v = np.vectorize(f)
Y_f = f_v(Y_pred)

with open(str(sys.argv[2]) + '/result_3.txt', 'w') as f:
    for y in Y_f:
        f.write(str(y[0]))
        f.write('\n')