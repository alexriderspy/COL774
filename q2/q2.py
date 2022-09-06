import sys
import pandas as pd
import numpy as np
import q2_b

theta,_ = q2_b.sgd(q2_b.arrX,q2_b.arrY,q2_b.m,0.001,100,0.9)

Y_pred = np.matmul(q2_b.testX,theta)

cost = q2_b.find_cost(q2_b.testX,q2_b.testY,theta,q2_b.testm,0)
print(theta)
print(cost)
with open(str(sys.argv[1]) + '/result_2.txt', 'w') as f:
    for y in Y_pred:
        f.write(str(y[0]))
        f.write('\n')