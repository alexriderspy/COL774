import sys
import pandas as pd
import numpy as np
import q2_b

theta,_ = q2_b.sgd(q2_b.arrX,q2_b.arrY,q2_b.m,0.001,100,0.01,100000)

Y_pred = np.matmul(q2_b.testX,theta)

with open('./result_2.txt', 'w') as f:
    for y in Y_pred:
        f.write(str(y[0]))
        f.write('\n')