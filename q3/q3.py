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
