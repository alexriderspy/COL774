import tqdm
import numpy as np
import pandas as pd
import random
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

num_iter = 500
m = len(x_train)
n = len(x_train[0])
r = 10
lr = 0.1
b = 100

hidden_layers= [100,100]
hidden_layers = [n] + hidden_layers + [r]

def sigmoid(x):
    x = np.clip(x,-700,np.inf)
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
weights = [np.random.randn(input_sizes[i],output_sizes[i]) for i in range(len(input_sizes))]
bias = [np.random.randn(1,output_sizes[i])/np.sqrt(input_sizes[i] + output_sizes[i]) for i in range(len(input_sizes))]

iter = 0
error = 100
while (iter <= num_iter) and (error > 0.01 or iter < 100) :
    iter += 1
    #lr = lr/np.sqrt(iter)
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
            if i == len(hidden_layers)-2:
                output = sigmoid(output)
            else:
                output = relu(output)
            o_ls.append(output)

        error += mse(y_true,output)

        output_error = np.sum(mse_prime(y_true,output),axis=0)

        #print(output_error.shape)
        for i in range(len(hidden_layers)-2,-1,-1):
            #output_error = output_error.reshape((len(output_error),))
            if i == len(hidden_layers)-2:
                output_error = output_error*sigmoid_prime(net_ls[i])
            else:
                output_error = output_error*relu_prime(net_ls[i])
            input_error = np.dot(output_error, weights[i].T)
            weights_error = np.dot(inputs[i].T,output_error)

            weights[i] -= lr*weights_error
            bias[i] -= lr*np.sum(output_error,axis=0)
            output_error = input_error

    error/=b

acc = 0

for k in range(m):
    output = x_train[k]
    for i in range(len(hidden_layers)-1):
        output = np.reshape(output, (1,-1))
        output = np.dot(output, weights[i]) + bias[i]
        if i == len(hidden_layers)-2:
            output = sigmoid(output)
        else:
            output = relu(output)

    index = np.argmax(output)
    acc += (index == y_trueval[k])
print('Accuracy of train under relu is ' + str(acc/m))

indices = []
acc=0.0
for k in range(len(x_test)):
    output = x_test[k]
    for i in range(len(hidden_layers)-1):
        output = np.reshape(output, (1,-1))
        output = np.dot(output, weights[i]) + bias[i]
        if i == len(hidden_layers)-2:
            output = sigmoid(output)
        else:
            output = relu(output)

    index = np.argmax(output)
    indices.append(index)
    acc += (index == y_test[k])
print('Accuracy of test under relu is ' + str(acc/len(x_test)))

y_test = y_test.flatten()
indices = np.array(indices)
print('confusion matrix under relu is ')
print(confusion_matrix(y_test,indices))

input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
weights = [np.random.randn(input_sizes[i],output_sizes[i]) for i in range(len(input_sizes))]
bias = [np.random.randn(1,output_sizes[i])/np.sqrt(input_sizes[i] + output_sizes[i]) for i in range(len(input_sizes))]

lr = 0.1

iter = 0
error = 100
while (iter <= num_iter) and (error > 0.05 or iter < 20) :
    iter += 1
    #lr = lr/np.sqrt(iter)
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

        error += mse(y_true,output)
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

    error/=b

acc = 0

for k in range(m):
    output = x_train[k]
    for i in range(len(hidden_layers)-1):
        output = np.reshape(output, (1,-1))
        output = np.dot(output, weights[i]) + bias[i]
        output = sigmoid(output)
    index = np.argmax(output)
    acc += (index == y_trueval[k])
print('Accuracy of train under sigmoid is ' + str(acc/m))

indices = []
acc=0.0
for k in range(len(x_test)):
    output = x_test[k]
    for i in range(len(hidden_layers)-1):
        output = np.reshape(output, (1,-1))
        output = np.dot(output, weights[i]) + bias[i]
        output = sigmoid(output)

    index = np.argmax(output)
    indices.append(index)
    acc += (index == y_test[k])
print('Accuracy of test under sigmoid is ' + str(acc/len(x_test)))

y_test = y_test.flatten()
indices = np.array(indices)
print('confusion matrix under sigmoid is ')
print(confusion_matrix(y_test,indices))