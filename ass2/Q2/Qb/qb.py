import os
import pickle
import sys
import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import pylab as pl

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])

file = os.path.join(train_path,'train_data.pickle')
test_file = os.path.join(test_path,'test_data.pickle')

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

with open(test_file, 'rb') as fo:
    test_dict = pickle.load(fo, encoding='bytes')

test_labels = test_dict['labels']
test_data = test_dict['data']

test_arrY = []
test_arrX = []

#3 -> -1, 4 -> 1

for i in range(len(test_labels)):
    if test_labels[i] == 3:
        test_arrX.append(test_data[i].flatten())
        test_arrY.append(-1)
    elif test_labels[i]==4:
        test_arrX.append(test_data[i].flatten())
        test_arrY.append(1)

test_m = len(test_arrX)
test_arrX = np.array(test_arrX).reshape(test_m,3072)
test_arrY = np.array(test_arrY).reshape(test_m,1)

test_arrX = np.multiply(test_arrX,1.0)

def gaussian_rbf(X,Y):
    return np.exp(-gamma*(np.linalg.norm(X-Y)**2))

C = 1.0

gamma = 0.001

K = np.zeros((m,m))

for i in range(m):
    for j in range(m):
        K[i,j] = gaussian_rbf(arrX[i],arrX[j])

P = cvxopt.matrix(np.outer(arrY,arrY)*K)
q = cvxopt.matrix(-1 * np.ones(m)) # q has shape m*1

tmp1 = np.diag(np.ones(m) * -1)
tmp2 = np.identity(m)
G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
tmp1 = np.zeros(m)
tmp2 = np.ones(m) * C
h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

A = cvxopt.matrix(1.0 * arrY, (1, m))
b = cvxopt.matrix(0.0)
# solve quadratic programming
cvxopt.solvers.options['show_progress'] = False
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
_lambda = np.ravel(solution['x'])

_wts_sv = _lambda.reshape((m,1))

#support vectors
sv = np.bitwise_and(_lambda>1e-5, _lambda<=C)
indices = np.arange(len(_lambda))[sv]
_lambda = _lambda[sv]
sv_x = arrX[sv]
sv_y = arrY[sv]

print('The number of support vectors are : ' + str(len(indices)))
print("Fraction of support vectors : " + str(len(indices)/m))

b=0.0

for i in range(len(_lambda)):
    b += sv_y[i]
    b -= np.sum(_lambda * sv_y * K[indices[i],sv])
b /= len(_lambda)

#predict
y_predict = np.zeros(len(arrX))
for i in range(len(arrX)):
    s = 0.0
    for ai, sv_yi, sv_xi in zip(_lambda, sv_y, sv_x):
        s += ai * sv_yi * gaussian_rbf(arrX[i], sv_xi)
    y_predict[i] = s

w = y_predict + b

results = np.sign(w)
results[results == 0] = 1

#accuracy
accu = 0.0
for i in range(len(results)):
    if results[i]==arrY[i]:
        accu += 1

score = accu/len(results)
print('accuracy of train data : ' + str(score))

y_predict = np.zeros(len(test_arrX))
for i in range(len(test_arrX)):
    s = 0.0
    for ai, sv_yi, sv_xi in zip(_lambda, sv_y, sv_x):
        s += ai * sv_yi * gaussian_rbf(test_arrX[i], sv_xi)
    y_predict[i] = s

w = y_predict + b

results = np.sign(w)
results[results == 0] = 1

#accuracy
accu = 0.0
for i in range(len(results)):
    if results[i]==test_arrY[i]:
        accu += 1

score = accu/len(results)
print('accuracy of test data : ' + str(score))

_lambda = _wts_sv
_lambda.sort()

array_images_top5 = np.append(_lambda,arrX,axis=1)
array_images_top5.sort()
array_images_top5 = array_images_top5[-5:]
array_images_top5 = np.delete(array_images_top5,0,1)

for i in range(len(array_images_top5)):
    array_image = array_images_top5[i]
    array_image = array_image.reshape((32,32,3)).astype('uint8')
    plt.imsave('Image_gaussian_' + str(i)+ '.png',array_image)
    plt.imshow(array_image, interpolation='nearest')
