import numpy as np

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
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

model = SVC()
ovo = OneVsOneClassifier(model)
ovo.fit(arrX, arrY)
yhat = ovo.predict(test_arrX)

table_a = np.zeros((6,6))
table_b = np.zeros((6,6))

for i in range(len(test_arrX)):
    table_b[yhat[i]+1][test_arrY[i]+1] += 1

table_b[0][1] = '0(A)'
table_b[0][2] = '1(A)'
table_b[0][3] = '2(A)'
table_b[0][4] = '3(A)'
table_b[0][5] = '4(A)'

table_b[1][0] = '0(P)'
table_b[2][0] = '1(P)'
table_b[3][0] = '2(P)'
table_b[4][0] = '3(P)'
table_b[5][0] = '4(P)' 

print(table_b)