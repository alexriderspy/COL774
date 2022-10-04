import numpy as np
import cvxopt
import sys,os,pickle
from functools import cmp_to_key

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])

file = os.path.join(train_path,'train_data.pickle')
test_file = os.path.join(test_path,'test_data.pickle')

with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

labels = dict['labels']
data = dict['data']

data = (data*1.0)/255.0

with open(test_file, 'rb') as fo:
    test_dict = pickle.load(fo, encoding='bytes')

test_labels = test_dict['labels']
test_data = test_dict['data']

test_arrY = []
test_arrX = []

#3 -> -1, 4 -> 1

for i in range(len(test_labels)):
    test_arrX.append(test_data[i].flatten())
    test_arrY.append(test_labels[i])

test_m = len(test_arrX)
test_arrX = np.array(test_arrX).reshape(test_m,3072)
test_arrY = np.ravel(test_arrY)

test_arrX = np.multiply(test_arrX,1.0)

test_arrX/=255.0

final_array = np.zeros((len(test_arrX),5)).tolist()

C = 1.0

gamma = 0.001

for l0 in range(5):
    for l1 in range(l0+1,5):
        arrY = []
        arrX = []

        #l0 -> -1, l1 -> 1
        
        for i in range(len(labels)):
            if labels[i] == l0:
                arrX.append(data[i].flatten())
                arrY.append(-1)
            elif labels[i]==l1:
                arrX.append(data[i].flatten())
                arrY.append(1)

        m = len(arrX)
        arrX = np.array(arrX).reshape(m,3072)
        arrY = np.array(arrY).reshape(m,1)

        arrX = np.multiply(arrX,1.0)

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

        #support vectors
        sv = np.bitwise_and(_lambda>1e-8, _lambda<=C)
        indices = np.arange(len(_lambda))[sv]
        num_sv = len(indices)
        _lambda = _lambda[sv].reshape((num_sv,1))
        sv_x = arrX[sv].reshape((num_sv,3072))
        sv_y = arrY[sv].reshape((num_sv,1))

        mul = _lambda*sv_y
        sv_X_norm = np.sum(sv_x ** 2, axis = -1)
        K = np.exp(-gamma * (X_norm[:,None] + sv_X_norm[None,:] - 2 * np.dot(arrX, sv_x.T)))

        y_predict = np.sum(mul * K.T,axis=0).reshape((m,1))

        b = np.sum(arrY-y_predict)
        b/=len(arrX)

        testX_norm = np.sum(test_arrX ** 2, axis = -1)
        K = np.exp(-gamma * (testX_norm[:,None] + sv_X_norm[None,:] - 2 * np.dot(test_arrX, sv_x.T)))

        testy_predict = np.sum(mul * K.T,axis=0).reshape((test_m,1))
        w = testy_predict + b

        score = np.sign(w)
        score[score == 0] = 1

        for i in range(len(score)):
            if (score[i]<0 and test_arrY[i]==l0):
                if final_array[i][l0]==0:
                    final_array[i][l0] = (1,abs(score[i]))
                else:
                    prev = final_array[i][l0][0]
                    final_array[i][l0] = (prev+1,abs(final_array[i][l0][1])+abs(score[i]))
            elif (score[i]>0 and test_arrY[i]==l1):
                if final_array[i][l1]==0:
                    final_array[i][l1] = (1,abs(score[i]))
                else:
                    prev = final_array[i][l1][0]
                    final_array[i][l1] = (prev+1,abs(final_array[i][l1][1])+abs(score[i]))

def compare(x,y):
    if x[0] == y[0]:
        return x[1] - y[1]
    else:
        return x[0] - y[0]

accu = 0.0

for i in range(len(test_arrX)):
    get_scores = final_array[i]
    for j in range(len(get_scores)):
        get_scores[j] = (final_array[i][j][0],final_array[i][j][1],j)
    sorted(get_scores, key = cmp_to_key(compare))
    y_pred = get_scores[-1][2]
    if y_pred == test_arrY[i]:
        accu += 1

accu/=len(test_arrX)
print("Accuracy of test data : " + str(accu))

