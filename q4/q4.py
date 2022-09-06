import sys
import numpy as np
import pandas as pd

dataX = pd.read_fwf(str(sys.argv[1]) + "/X.dat",header=None)
dataY = pd.read_fwf(str(sys.argv[1]) + "/Y.dat",header=None)

testdataX = pd.read_fwf(str(sys.argv[2]) + "/X.dat", header = None)

arrX = dataX.to_numpy(dtype=float)
arrY = dataY.to_numpy()
testX = dataX.to_numpy(dtype=float)

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

mean_testX = np.mean(testX,axis=0)
std_testX = np.std(testX,axis=0)
testX = (testX - meanX)/stdX

testm = testX.shape[0]
z = np.ones((testm,1))
testX = np.append(testX,z,axis=1)
testX[:, [1,2]] = testX[:,[2,1]]
testX[:, [0,1]] = testX[:,[1,0]]

testX.reshape((testm,3))

m = arrX.shape[0]
n = arrX.shape[1]

arrX.reshape((m,2))
arrY.reshape((m,1))

#parameters of the 2 gaussians 
# mu0 for alaska since 0 is alaska

phi = np.sum(arrY)/m
mu1 = np.sum(arrX * arrY,axis=0).reshape((2,1))/np.sum(arrY)

mu0 = np.sum(arrX * (1-arrY),axis=0).reshape((2,1))/np.sum(1-arrY)

sigma = (1/m) * np.matmul((arrY*((arrX.T - mu1).reshape((m,2)))+ (1-arrY)*((arrX.T - mu0).reshape((m,2)))).T,(arrY*((arrX.T - mu1).reshape((m,2)))+ (1-arrY)*((arrX.T - mu0).reshape((m,2))))).reshape((2,2))

print(mu0)
print(mu1)
print(sigma)

inv_sigma = np.linalg.pinv(sigma)

a=inv_sigma[0][0]
b=inv_sigma[0][1]
c=inv_sigma[1][0]
d=inv_sigma[1][1]

mu01 = mu0[0][0]
mu02 = mu0[1][0]

mu11 = mu1[0][0]
mu12 = mu1[1][0]

intercept = 2*np.log((1-phi)/phi) + a*(mu01 + mu11)*(mu11-mu01) + d*(mu02+mu12)*(mu12-mu02) - (b+c)*(mu01*mu02 - mu11*mu12)

X1_min = -0.5
X1_max = 0.5

theta0 = -intercept
theta1 = 2*a - mu02*(b+c) + mu12*(b+c)
theta2 = 2*d - mu01*(b+c) + mu11*(b+c)

print(theta0)
print(theta1)
print(theta2)

X = np.linspace(X1_min,X1_max,10)
Y = (-theta0 - X*(theta1))/(theta2)

theta = np.array([theta0,theta1,theta2]).reshape((3,1))

Y_pred = 1/(1+np.exp(-np.matmul(testX,theta)))

def f(x):
    if x > 0.5:
        return "Canada"
    else:
        return "Alaska"

f_v = np.vectorize(f)
Y_f = f_v(Y_pred)

with open(str(sys.argv[2]) + '/result_4.txt', 'w') as f:
    for y in Y_f:
        f.write(str(y[0]))
        f.write('\n')