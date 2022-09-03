import sys
import pandas as pd
import numpy as np
import q2_b

testdata = pd.read_csv(str(sys.argv[1]) + "/X.csv")

testdata = testdata.to_numpy()
print(testdata)

arrY = testdata[:,2]
arrX = np.delete(testdata, 2,1)

m = arrX.shape[0]
n = arrX.shape[1]
z = np.ones((m,1))

arrX = np.append(arrX,z,axis=1)
arrX[:, [0,2]] = arrX[:,[2,0]]
arrX[:, [1,2]] = arrX[:,[2,1]]

arrX.reshape((m,3))
arrY.reshape((m,1))

print(arrX)
print(arrY)

theta = q2_b.sgd(arrX,arrY,m,0.001,1000)
Y_pred = np.matmul(arrX,theta)

cost = q2_b.find_cost(arrX,arrY,theta,1000,0)
print(cost)

with open(str(sys.argv[1]) + '/result_2.txt', 'w') as f:
    for y in Y_pred:
        f.write(str(y[0]))
        f.write('\n')