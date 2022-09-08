import numpy as np
import pandas as pd
import sys
import q2_a

testdata = pd.read_csv(str(sys.argv[1]) + "/X.csv",header=None)

testdata = testdata.to_numpy()

# testY = testdata[:,2]
# testX = np.delete(testdata, 2,1)

testX = testdata
testm = testX.shape[0]
n = testX.shape[1]
z = np.ones((testm,1))

testX = np.append(testX,z,axis=1)
testX[:, [0,2]] = testX[:,[2,0]]
testX[:, [1,2]] = testX[:,[2,1]]

testX.reshape((testm,3))
#testY.reshape((testm,1))

arrX = q2_a.arrX
arrY = q2_a.arrY

m = q2_a.m

arrX = np.append(arrX,arrY,axis=1)
np.random.shuffle(arrX)
arrY = arrX[:,3]
arrX = np.delete(arrX,3,1)

def find_cost(arrX,arrY,theta,b,i):
  x = (i*b)%m
  cost_fn = (1/(2*b))*np.sum((arrY[x:b+x].reshape((b,1)) - np.matmul(arrX[x:b+x,:],theta))**2)
  return cost_fn

log = {}

def sgd(arrX,arrY,m,learning_rate,b,stop,max_iter):

  log["batch_size"] = b
  log["epochs"] = 0
  log["theta"] = []
  log["error"] = []
  theta = np.zeros(3).reshape((3,1))
  i=0
  iter = 0
  cost = 100
  while (cost > stop or iter < 1000) and iter < max_iter:
    x = (i*b)%m
    log["theta"].append(theta)
    theta = theta + (learning_rate/b) * np.sum(np.multiply(arrX[x:(b+x),:],arrY[x:(b+x)].reshape((b,1)) - np.matmul(arrX[x:(b+x),:],theta)).T,axis=1).reshape((3,1))
    iter += 1
    cost = find_cost(arrX,arrY,theta,min(b,1000),(i-1000+m)%m)
    log["error"].append(cost)
    i = (i+1)%m
  log["epochs"] = iter
  return theta,log