from sklearn import svm
import numpy as np
import pickle,sys,os,cvxopt

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
arrY = np.array(arrY)

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

trainedsvm_linear = svm.SVC(kernel = 'linear').fit(arrX, arrY)

print("Weights in case of linear sklearn model : ")
print(trainedsvm_linear.coef_)
print("Bias in case of linear sklearn model : ")
print(trainedsvm_linear.intercept_)
score = trainedsvm_linear.score(test_arrX,test_arrY)
support_vector_indices = trainedsvm_linear.support_
print("Accuracy of test data in linear sklearn : " + str(score))

C = 1.0

arrY = arrY.reshape((m,1))
K = (arrX * arrY).T
P = cvxopt.matrix(K.T.dot(K)) # P has shape m*m
q = cvxopt.matrix(-1 * np.ones(m)) # q has shape m*1
G = cvxopt.matrix(np.concatenate((-1*np.identity(m), np.identity(m)), axis=0))
h = cvxopt.matrix(np.concatenate((np.zeros(m), C*np.ones(m)), axis=0))
A = cvxopt.matrix(1.0 * arrY, (1, m))
b = cvxopt.matrix(0.0)

cvxopt.solvers.options['show_progress'] = False
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
_lambda = np.ravel(solution['x']).reshape(m,1)

S = np.where((_lambda > 1e-5) & (_lambda <= C))[0]

print("Number of support vectors that match in linear kernel case: ")
print(len(np.intersect1d(np.ravel(S),np.ravel(support_vector_indices))))

arrY = np.ravel(arrY)
trainedsvm_gaussian = svm.SVC(kernel = 'rbf',gamma = 0.001).fit(arrX, arrY)

score = trainedsvm_gaussian.score(test_arrX,test_arrY)
support_vector_indices = trainedsvm_gaussian.support_
print("Accuracy of test data in gaussian sklearn : " + str(score))

C = 1.0

gamma = 0.001

X_norm = np.sum(arrX ** 2, axis = -1)
K = np.exp(-gamma * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(arrX, arrX.T)))

arrY = arrY.reshape((m,1))
P = cvxopt.matrix(np.outer(arrY,arrY)  * K)
q = cvxopt.matrix(-1 * np.ones(m)) # q has shape m*1
G = cvxopt.matrix(np.concatenate((-1*np.identity(m), np.identity(m)), axis=0))
h = cvxopt.matrix(np.concatenate((np.zeros(m), C*np.ones(m)), axis=0))
A = cvxopt.matrix(1.0 * arrY, (1, m))
b = cvxopt.matrix(0.0)
# solve quadratic programming
cvxopt.solvers.options['show_progress'] = False
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
_lambda = np.ravel(solution['x'])

#support vectors
sv = np.bitwise_and(_lambda>1e-5, _lambda<=C)
indices = np.arange(len(_lambda))[sv]
num_sv = len(indices)

print("Number of support vectors that match in gaussian kernel case: ")
print(len(np.intersect1d(np.ravel(indices),np.ravel(support_vector_indices))))