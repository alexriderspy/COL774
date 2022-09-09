import sys
import pandas as pd
import numpy as np
import q1_a

testdataX = pd.read_csv(str(sys.argv[2]) + "/X.csv", header = None)
testX = testdataX.to_numpy()
m = testX.shape[0]
n = testX.shape[1]
z = np.ones((m,1))

testX = (testX - q1_a.meanX)/q1_a.stdX

testX = np.append(testX,z,axis=1)
testX[:, [0,1]] = testX[:,[1,0]]

testX.reshape((m,2))

Y_pred = np.matmul(testX,q1_a.theta)

with open('./result_1.txt', 'w') as f:
    for y in Y_pred:
        f.write(str(y[0]))
        f.write('\n')