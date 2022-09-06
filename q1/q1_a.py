import sys
import pandas as pd
import numpy as np

dataX = pd.read_csv(str(sys.argv[1]) + "/X.csv",header=None)
dataY = pd.read_csv(str(sys.argv[1]) + "/Y.csv",header=None)

arrX = dataX.to_numpy()
arrY = dataY.to_numpy()

meanX = np.mean(arrX)
stdX = np.std(arrX)
arrX = (arrX - meanX)/stdX

m = arrX.shape[0]
n = arrX.shape[1]
z = np.ones((m,1))

arrX = np.append(arrX,z,axis=1)
arrX[:, [0,1]] = arrX[:,[1,0]]

arrX.reshape((m,2))
arrY.reshape((m,1))

def find_cost(arrX,arrY,theta):
  cost_fn = (1/(2*m))*np.sum((arrY - np.matmul(arrX,theta))**2)
  return cost_fn

eta = 0.1

log = {"alpha": eta,
        "epochs": 0,
        "train_loss": [],
        "theta": []}

def linear_regression(learning_rate):
  log["alpha"] = learning_rate
  theta = np.zeros(2).reshape((2,1))
  cost_delta = 1000
  cost = -1
  log["theta"] = []
  log["train_loss"] = []
  while cost_delta > 1e-9:
    theta = theta + (learning_rate/m) * np.sum((arrX * (arrY - np.matmul(arrX,theta))).T,axis=1).reshape((2,1))
    log["theta"].append(theta.flatten())
  
    newcost = find_cost(arrX,arrY,theta)
    log["train_loss"].append(newcost)

    cost_delta = abs(newcost - cost)
    cost = newcost
  return log,theta

log,theta = linear_regression(0.1)
Y_pred = np.matmul(arrX,theta)
print(theta)