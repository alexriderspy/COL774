import tqdm
import numpy as np
import pandas as pd
import random
from keras.utils import np_utils
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier

train_path = 'fmnist_train.csv'
test_path = 'fmnist_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

x_train = train_data.to_numpy()
random.shuffle(x_train)
y_trueval = x_train[:,-1]
y_train = x_train[:,-1]

x_test = test_data.to_numpy()
y_test = test_data.to_numpy()[:,-1]

x_train = np.delete(x_train,784,axis=1)
x_test = np.delete(x_test,784,axis=1)

x_train = x_train.astype('float64')
x_train /= 255
y_train = np_utils.to_categorical(y_train)

x_test = x_test.astype('float64')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

clf = MLPClassifier(random_state=1, max_iter=300, solver='sgd', activation='relu', hidden_layer_sizes=[50,50,50,50]).fit(x_train, y_train)
acc_train = clf.score(x_train,y_train)
print("Training accuracy is " + str(acc_train))
acc_test = clf.score(x_test, y_test)
print("Test accuracy is " + str(acc_test))