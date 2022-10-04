import random
import numpy as np

from sklearn import svm
import sys,os,pickle
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
    arrX.append(data[i].flatten())
    arrY.append(labels[i])

arrX = np.append(arrX,arrY,axis=1)
random.shuffle(arrX)
arrY = arrX[:,3072]
arrX = np.delete(arrX,3072,1)

m = len(arrX)
arrX = np.array(arrX).reshape(m,3072)
arrY = np.ravel(arrY)

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
    test_arrX.append(test_data[i].flatten())
    test_arrY.append(test_labels[i])

test_m = len(test_arrX)
test_arrX = np.array(test_arrX).reshape(test_m,3072)
test_arrY = np.ravel(test_arrY)

test_arrX = np.multiply(test_arrX,1.0)

test_arrX/=255.0

C_values = [1e-5,1e-3,1,5,10]
gamma = 0.001
length = len(arrX)//5

val_accu = [0 for _ in range(5)]
test_accu = [0 for _ in range(5)]

iter = 0
for C in C_values:
    i=0
    maxi = -10.0
    avg_val = 0.0
    trained = None
    while(i<length*5):
        valX = arrX[i:i+length]
        valY = arrY[i:i+length]
        trainX = np.append(arrX[0:i],arrX[i+length:],axis=0)
        trainY = np.append(arrY[0:i],arrY[i+length:],axis=0)
        model = svm.SVC(kernel = 'rbf',gamma = 0.001,decision_function_shape='ovo',C=C)
        model.fit(trainX, trainY)
        yhat = model.predict(valX)
        
        accu = np.sum(yhat == valY)
        accu/= len(valY)
        #print(accu)
        avg_val += accu
        if accu>maxi:
            maxi = accu
            trained = model
        i+=length
    
    val_accu[iter] = avg_val/5
    yhat = trained.predict(test_arrX)
    accu = np.sum(yhat == test_arrY)
    accu /= len(test_arrY)
    test_accu[iter]=accu
    iter += 1

print("Validation accuracy for each value of C : ")
print(val_accu)
print("Test accuracy for each value of C : ")
print(test_accu)

plt.xlabel('C')
plt.ylabel('validation accuracy')
plt.plot(C_values,val_accu)
plt.savefig('val_accuracy vs C')

plt.figure()
plt.xlabel('C')
plt.ylabel('test accuracy')
plt.plot(C_values,test_accu)
plt.savefig('test_accuracy vs C')