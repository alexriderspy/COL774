import numpy as np
import q2_a

arrX = q2_a.arrX
arrY = q2_a.arrY

m = q2_a.m

arrX = np.append(arrX,arrY,axis=1)
np.random.shuffle(arrX)
arrY = arrX[:,3]
arrX = np.delete(arrX,3,1)

def find_cost(arrX,arrY,theta,b,i):
  cost_fn = (1/(2*b))*np.sum((arrY[i:b+i].reshape((b,1)) - np.matmul(arrX[i:b+i,:],theta))**2)
  return cost_fn

def amt_converge(orig_theta, theta):
  return np.sum((orig_theta-theta)**2)

log = {}

def sgd(arrX,arrY,m,learning_rate,b):

  log["batch_size"] = b
  log["epochs"] = 0
  log["theta"] = []
  theta = np.zeros(3).reshape((3,1))
  print(find_cost(arrX,arrY,theta,m,0))
  i=0
  max_iter = 5000000
  iter = 0
  while amt_converge(np.array([3,1,2]).reshape((3,1)),theta) > 0.01 and iter < max_iter:
    x = (i*b)%m
    log["theta"].append(theta)
    theta = theta + (learning_rate/b) * np.sum(np.multiply(arrX[x:(b+x),:],arrY[x:(b+x)].reshape((b,1)) - np.matmul(arrX[x:(b+x),:],theta)).T,axis=1).reshape((3,1))
    iter += 1
    i = (i+1)%m
  log["epochs"] = iter
  return theta,log

logs = {}
for r in [1,100,10000]:
  theta,log = sgd(arrX,arrY,m,0.001,r)
  logs[r] = log
  print(theta)
  print(find_cost(arrX,arrY,theta,m,0))

best_batch = 10000