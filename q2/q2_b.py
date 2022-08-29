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

def sgd(learning_rate,b):
  theta = np.zeros(3).reshape((3,1))
  print(find_cost(arrX,arrY,theta,m,0))
  for i in range(m):
    x = (i*b)%m
    theta = theta + (learning_rate/b) * np.sum(np.multiply(arrX[x:(b+x),:],arrY[x:(b+x)].reshape((b,1)) - np.matmul(arrX[x:(b+x),:],theta)).T,axis=1).reshape((3,1))
  return theta

theta = sgd(0.001,100)
print(theta)
print(find_cost(arrX,arrY,theta,m,0))
