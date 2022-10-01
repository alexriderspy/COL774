import os
import pickle
import sys
import numpy as np
import cvxopt

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])

file = os.join(train_path,'train_data.pickle')

with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

labels = dict['labels']
data = dict['data']

arrY = []
arrX = []

#3 -> -1, 4 -> 1

for i in range(len(labels)):
    if labels[i] == 3:
        arrX.append(data[i].flatten())
        arrY.append(-1)
    elif labels[i]==4:
        arrX.append(data[i].flatten())
        arrY.append(1)

m = len(arrX)
arrX = np.array(arrX).reshape(m,3072)
arrY = np.array(arrY).reshape(m,1)

arrX = np.multiply(arrX,1.0)

C = 1.0

# compute inputs for cvxopt solver
K = (arrX * arrY).T
P = cvxopt.matrix(K.T.dot(K)) # P has shape m*m
q = cvxopt.matrix(-1 * np.ones(m)) # q has shape m*1
G = cvxopt.matrix(np.concatenate((-1*np.identity(m), np.identity(m)), axis=0))
h = cvxopt.matrix(np.concatenate((np.zeros(m), C*np.ones(m)), axis=0))
A = cvxopt.matrix(1.0 * arrY, (1, m))
b = cvxopt.matrix(0.0)
# solve quadratic programming
cvxopt.solvers.options['show_progress'] = False
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
_lambda = np.ravel(solution['x'])
print(_lambda)

S = np.where((_lambda > 1e-10) & (_lambda <= C))[0]
print('The number of support vectors are : ' + str(len(S)))
print("Fraction of support vectors : " + str(len(S))/m)

w = K[:, S].dot(_lambda[S])

M = np.where((_lambda > 1e-10) & (_lambda < C))[0]
b = np.mean(arrY[M] - arrX[M, :].dot(w))

results = np.sign(arrX.dot(w) +b)
results[results == 0] = 1

accu = 0
for i in range(len(results)):
    if results[i]==arrY[i]:
        accu += 1

score = accu/len(results)
print(score)