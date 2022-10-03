from sklearn.svm import SVC
import numpy as np
import sys,os,pickle

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
    arrX.append(data[i].flatten())
    arrY.append(labels[i])

m = len(arrX)
arrX = np.array(arrX).reshape(m,3072)
arrY = np.ravel(arrY)

arrX = np.multiply(arrX,1.0)

meanX = np.mean(arrX)
stdX = np.std(arrX)
arrX/=255.0

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

model = SVC(kernel = 'rbf',gamma = 0.001,decision_function_shape='ovo')
model.fit(arrX, arrY)
yhat = model.predict(test_arrX)

accu = np.sum(yhat == test_arrY)
accu/= len(test_arrY)
print("Accuracy of test data : " + str(accu))
#0.638 is accuracy