from sklearn import svm
import numpy as np
import pickle

file = '../part2_data/train_data.pickle'

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

trainedsvm_linear = svm.SVC(kernel = 'linear').fit(arrX, arrY)
predictionsvm = trainedsvm_linear.predict(arrX)

score = trainedsvm_linear.score(arrX,arrY)
print(score)
#1.0

trainedsvm_gaussian = svm.SVC(kernel = 'rbf').fit(arrX, arrY)
predictionsvm = trainedsvm_gaussian.predict(arrX)

score = trainedsvm_gaussian.score(arrX,arrY)
print(score)
#0.904