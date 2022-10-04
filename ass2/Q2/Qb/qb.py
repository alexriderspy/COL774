import os
import pickle
import sys
import numpy as np
import cvxopt
import matplotlib.pyplot as plt

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

arrX/=255.0

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

test_arrX/=255.0

C = 1.0

gamma = 0.001

def gaussian_rbf(X,Y):
    X = np.ravel(X)
    global gamma
    return np.exp(-gamma*(np.linalg.norm(Y-X, axis=1)**2))

X_norm = np.sum(arrX ** 2, axis = -1)
K = np.exp(-gamma * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(arrX, arrX.T)))

P = cvxopt.matrix(np.outer(arrY,arrY)  * K)
q = cvxopt.matrix(-1 * np.ones(m)) # q has shape m*1
G = cvxopt.matrix(np.concatenate((-1*np.identity(m), np.identity(m)), axis=0))
h = cvxopt.matrix(np.concatenate((np.zeros(m), C*np.ones(m)), axis=0))
A = cvxopt.matrix(1.0 * arrY, (1, m))
b = cvxopt.matrix(0.0)
# solve quadratic programming
#cvxopt.solvers.options['show_progress'] = False
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
_lambda = np.ravel(solution['x'])

_wts_sv = _lambda.reshape((m,1))

#support vectors
sv = np.bitwise_and(_lambda>1e-8, _lambda<=C)
indices = np.arange(len(_lambda))[sv]
num_sv = len(indices)
_lambda = _lambda[sv].reshape((num_sv,1))
sv_x = arrX[sv].reshape((num_sv,3072))
sv_y = arrY[sv].reshape((num_sv,1))

print('The number of support vectors are : ' + str(num_sv))
print("Fraction of support vectors : " + str(num_sv/m))

#predict
y_predict = np.zeros((m,1))
testy_predict = np.zeros((test_m,1))

mul = _lambda*sv_y

for i in range(m):
    y_predict[i] = np.sum(mul*gaussian_rbf(arrX[i],sv_x).reshape((num_sv,1)))
    if i<test_m:
        testy_predict[i] = np.sum(mul*gaussian_rbf(test_arrX[i],sv_x).reshape((num_sv,1)))

b = np.sum(arrY-y_predict)
b/=len(arrX)

#testy_predict = np.sum(mul*gaussian_rbf(test_arrX,sv_x).reshape((num_sv,1)),axis=0).reshape((num_sv,1))

w = testy_predict + b

results = np.sign(w)
results[results == 0] = 1

score = (np.sum(results==test_arrY))/len(results)
print('accuracy of test data : ' + str(score))

_lambda = _wts_sv

indices = np.arange(m).reshape((m,1))
_lambda = np.append(_lambda,indices,axis=1)
_lambda.sort()

array_images_top5 = _lambda[-5:]

for i in range(len(array_images_top5)):
    array_image = array_images_top5[i][1]
    img = arrX[int(array_image)]
    img*=255.0
    img = img.reshape((32,32,3)).astype('uint8')
    plt.imsave('Image_gaussian_' + str(i)+ '.png',img)
