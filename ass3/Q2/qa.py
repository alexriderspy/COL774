import numpy as np
import pandas as pd
import random
from keras.utils import np_utils

train_path = 'fmnist_train.csv'
test_path = 'fmnist_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

x_train = train_data.to_numpy()
random.shuffle(x_train)
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

hidden_layers = [100,50]
lr = 0.1
num_iter = 100
m = len(x_train)
n = len(x_train[0])
r = 10
lr = 0.1
b = 100

pipeline = []
hidden_layers = [n] + hidden_layers + [r]

input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
weights = [np.random.randn(input_sizes[i],output_sizes[i]) for i in range(len(input_sizes))]
bias = [np.random.randn(1,output_sizes[i])/np.sqrt(input_sizes[i] + output_sizes[i]) for i in range(len(input_sizes))]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

for iter in range(num_iter):
    error = 0
    for k in range(0,m,b):
        x = x_train[k:k+b]
        y_true = y_train[k:k+b]
        
        output = x
        #forward
        
        net_ls = []
        o_ls = []
        inputs=[]
        for i in range(len(hidden_layers)-1):
            #output = np.reshape(output, (1,-1))
            inputs.append(output)
            output = np.dot(output, weights[i]) + bias[i]
            net_ls.append(output)
            output = sigmoid(output)
            o_ls.append(output)
        
        #error += mse(y_true,output)
        
        output_error = np.sum(mse_prime(y_true,output),axis=0)
        
        #print(output_error.shape)
        for i in range(len(hidden_layers)-2,-1,-1):
            #output_error = output_error.reshape((len(output_error),))
            output_error = output_error*sigmoid_prime(net_ls[i])
            input_error = np.dot(output_error, weights[i].T)
            weights_error = np.dot(inputs[i].T,output_error)
            
            weights[i] -= lr*weights_error
            bias[i] -= lr*np.sum(output_error,axis=0)
            output_error = input_error

error = 0

for x,y_true in zip(x_train,y_train):
    output = x
    for i in range(len(hidden_layers)-1):
        output = np.reshape(output, (1,-1))
        inputs.append(output)
        output = np.dot(output, weights[i]) + bias[i]
        net_ls.append(output)
        output = sigmoid(output)
        o_ls.append(output)

    error += mse(y_true,output)
print(error/m)