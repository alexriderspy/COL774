import sys
import numpy as np
import pandas as pd

dataX = pd.read_csv(str(sys.argv[1]) + "/X.csv",header=None)
dataY = pd.read_csv(str(sys.argv[1]) + "/Y.csv",header=None)

testdataX = pd.read_csv(str(sys.argv[2]) + "/X.csv", header = None)

arrX = dataX.to_numpy()
arrY = dataY.to_numpy()
testX = testdataX.to_numpy()

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

testX.reshape((testm,2))

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

# print(mu0)
# print(mu1)
# print(sigma)

inv_sigma = np.linalg.pinv(sigma)

a=inv_sigma[0][0]
b=inv_sigma[0][1]
c=inv_sigma[1][0]
d=inv_sigma[1][1]

mu01 = mu0[0][0]
mu02 = mu0[1][0]

mu11 = mu1[0][0]
mu12 = mu1[1][0]

sigma1 = (1/np.sum(arrY)) * np.matmul((arrY*((arrX.T - mu1).reshape((m,2)))).T,(arrY*((arrX.T - mu1).reshape((m,2))))).reshape((2,2))
sigma0 = (1/np.sum(1-arrY)) * np.matmul(((1-arrY)*((arrX.T - mu0).reshape((m,2)))).T,((1-arrY)*((arrX.T - mu0).reshape((m,2))))).reshape((2,2))

inv_sigma0 = np.linalg.pinv(sigma0)
inv_sigma1 = np.linalg.pinv(sigma1)

a1=inv_sigma0[0][0]
b1=inv_sigma0[0][1]
c1=inv_sigma0[1][0]
d1=inv_sigma0[1][1]

a2=inv_sigma1[0][0]
b2=inv_sigma1[0][1]
c2=inv_sigma1[1][0]
d2=inv_sigma1[1][1]

intercept = 2*np.log((1-phi)/phi) + a*(mu01 + mu11)*(mu11-mu01) + d*(mu02+mu12)*(mu12-mu02) - (b+c)*(mu01*mu02 - mu11*mu12)

X1_min = -0.5
X1_max = 0.5

theta0 = -intercept
theta1 = 2*a - mu02*(b+c) + mu12*(b+c)
theta2 = 2*d - mu01*(b+c) + mu11*(b+c)

# print(theta0)
# print(theta1)
# print(theta2)

X = np.linspace(X1_min,X1_max,10)
Y = (-theta0 - X*(theta1))/(theta2)

def func(x1,x2):
    z = a1*(x1*x1+mu01*mu01-2*x1*mu01) + (b1+c1)*(x2*x1 - mu02*x1-mu01*x2+mu01*mu02) + d1*(x2*x2 + mu02*mu02-2*x2*mu02)-((2*np.log((1-phi)/phi))+a2*(x1*x1+mu11*mu11-2*x1*mu11) + (b2+c2)*(x1*x2-x1*mu12-x2*mu11+mu11*mu12) + d2*(x2*x2+mu12*mu12-2*x2*mu12))
    if z > 0:
        return "Canada"
    else:
        return "Alaska"

with open('./result_4.txt', 'w') as f:
    for x in testX:
        y = func(x[0],x[1])
        f.write(str(y))
        f.write('\n')