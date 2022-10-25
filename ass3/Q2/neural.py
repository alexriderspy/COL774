import sys

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])
output_path = str(sys.argv[3])
q_part = str(sys.argv[4])

out = ''

if q_part == 'a':
    
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import confusion_matrix

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.to_numpy()
    random.shuffle(x_train)
    y_trueval = x_train[:,-1]
    y_train = x_train[:,-1]
    y_train = y_train.reshape((-1,1))
    x_test = test_data.to_numpy()
    y_test = test_data.to_numpy()[:,-1]

    x_train = np.delete(x_train,784,axis=1)
    x_test = np.delete(x_test,784,axis=1)

    x_train = x_train.astype('float64')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.astype('float64')
    x_test /= 255

    hidden_layers_list = []
    lr = 0.1
    num_iter = 1000
    m = len(x_train)
    n = len(x_train[0])
    r = 10
    b = 100

    for x in [100]:
        hidden_layers = [n] + [x] + [x] + [r]
        hidden_layers_list.append(hidden_layers)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        return x*(1-x)

    def mse(y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))/(2*len(y_true))

    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size


    def relu(z):
        return np.maximum(0, z)
    def relu_prime(z):
        return 1. * (z > 0)

    for hidden_layers in   (hidden_layers_list):
        start_time = time.time()
        iter = 0
        error = 100
        input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
        output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
        weights = [np.random.randn(output_sizes[i],input_sizes[i])*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        bias = [np.random.randn(output_sizes[i],1)*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        prev_cost = 1e9
        curr_cost = 0
        while (iter < num_iter) :
            prev_cost = curr_cost
            iter += 1
            rate = lr
            rate = lr/iter**0.5
            error = 0
            for k in range(0,m,b):
                x = x_train[k:k+b]
                y_true = y_train[k:k+b]
                
                output = x
                o_ls = [output]
                inputs=[]
                for i in range(len(hidden_layers)-1):
                    #output = np.reshape(output, (1,-1))
                    inputs.append(output)
                    output = np.dot(output, weights[i].T) + bias[i].T
                    output = sigmoid(output)
                    o_ls.append(output)
                curr_cost += mse(y_true,output)
                
                output_error = (y_true-output)*(sigmoid_prime(output))
                deltas = [output_error]
                for i in range(len(hidden_layers)-2,0,-1):
                    deltas.append(np.matmul(deltas[-1], weights[i])*sigmoid_prime(o_ls[i]))

                deltas.reverse()
                
                for i in range(len(hidden_layers)-1):
                    weights[i] += rate*np.matmul(deltas[i].T,o_ls[i])/b
                    bias[i] += rate*np.sum(deltas[i],axis=0,keepdims=True).T/b
            curr_cost/=(m/b)
            
        time_taken = time.time() - start_time
        acc = 0

        for k in range(m):
            output = x_train[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)
            index = np.argmax(output)
            acc += (index == y_trueval[k])
        out += ('Accuracy of train is ' + str(acc/m))

        indices = []
        acc=0.0
        for k in range(len(x_test)):
            output = x_test[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)

            index = np.argmax(output)
            indices.append(index)
            acc += (index == y_test[k])
        out += ('Accuracy of test is ' + str(acc/len(x_test)))
    output_file = open(output_path + '/a.txt','w')
    output_file.write(out)
elif q_part == 'b':
     
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import confusion_matrix

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.to_numpy()
    random.shuffle(x_train)
    y_trueval = x_train[:,-1]
    y_train = x_train[:,-1]
    y_train = y_train.reshape((-1,1))
    x_test = test_data.to_numpy()
    y_test = test_data.to_numpy()[:,-1]

    x_train = np.delete(x_train,784,axis=1)
    x_test = np.delete(x_test,784,axis=1)

    x_train = x_train.astype('float64')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)
    x_test = x_test.astype('float64')
    x_test /= 255

    hidden_layers_list = []

    lr = 0.1
    num_iter = 1000
    m = len(x_train)
    n = len(x_train[0])
    r = 10
    b = 100

    for x in [5,10,15,20,25]:
        hidden_layers = [n] + [x] + [r]
        hidden_layers_list.append(hidden_layers)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        return x*(1-x)

    def mse(y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))/(2*len(y_true))

    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size

    train_accuracies = []
    test_accuracies = []
    times = []
    for hidden_layers in   (hidden_layers_list):
        start_time = time.time()
        iter = 0
        error = 100
        input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
        output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
        weights = [np.random.randn(output_sizes[i],input_sizes[i])*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        bias = [np.random.randn(output_sizes[i],1)*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        prev_cost = 1e9
        curr_cost = 0
        while (iter < num_iter) :
            prev_cost = curr_cost
            iter += 1
            rate = lr
            #rate = lr/iter**0.5
            error = 0
            for k in range(0,m,b):
                x = x_train[k:k+b]
                y_true = y_train[k:k+b]
                
                output = x
                o_ls = [output]
                inputs=[]
                for i in range(len(hidden_layers)-1):
                    #output = np.reshape(output, (1,-1))
                    inputs.append(output)
                    output = np.dot(output, weights[i].T) + bias[i].T
                    output = sigmoid(output)
                    o_ls.append(output)
                
                curr_cost += mse(y_true,output)
                output_error = (y_true-output)*(sigmoid_prime(output))
                deltas = [output_error]
                for i in range(len(hidden_layers)-2,0,-1):
                    deltas.append(np.matmul(deltas[-1], weights[i])*sigmoid_prime(o_ls[i]))

                deltas.reverse()
                
                for i in range(len(hidden_layers)-1):
                    weights[i] += rate*np.matmul(deltas[i].T,o_ls[i])/b
                    bias[i] += rate*np.sum(deltas[i],axis=0,keepdims=True).T/b
            curr_cost/=(m/b)
            
        time_taken = time.time() - start_time
        acc = 0

        for k in range(m):
            output = x_train[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)

            index = np.argmax(output)
            #out += (index)
            #out += (y_true[k])
            acc += (index == y_trueval[k])
        out += ('Accuracy of train is ' + str(acc/m))
        train_accuracies.append(acc/m)

        indices = []
        acc=0.0
        for k in range(len(x_test)):
            output = x_test[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)

            index = np.argmax(output)
            indices.append(index)
            acc += (index == y_test[k])
        out += ('Accuracy of test is ' + str(acc/len(x_test)))
        test_accuracies.append(acc/len(x_test))
        
        y_test = y_test.flatten()
        indices = np.array(indices)
        out += (confusion_matrix(y_test,indices))
        
        times.append(time_taken)

    plt.plot([5,10,15,20,25],train_accuracies)
    plt.xlabel('Hidden layers')
    plt.ylabel('Train accuracies')
    plt.xlabel('Hidden units')
    plt.ylabel('Test accuracies')
    plt.plot([5,10,15,20,25],test_accuracies)
    plt.xlabel('Hidden layers')
    plt.ylabel('Times taken')
    plt.plot([5,10,15,20,25],times)
    output_file = open(output_path + '/b.txt','w')
    output_file.write(out)

elif q_part == 'c':
     
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import confusion_matrix

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.to_numpy()
    random.shuffle(x_train)
    y_trueval = x_train[:,-1]
    y_train = x_train[:,-1]
    y_train = y_train.reshape((-1,1))
    x_test = test_data.to_numpy()
    y_test = test_data.to_numpy()[:,-1]

    x_train = np.delete(x_train,784,axis=1)
    x_test = np.delete(x_test,784,axis=1)

    x_train = x_train.astype('float64')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)
    x_test = x_test.astype('float64')
    x_test /= 255

    hidden_layers_list = []

    lr = 0.1
    num_iter = 1000
    m = len(x_train)
    n = len(x_train[0])
    r = 10
    b = 100

    for x in [5,10,15,20,25]:
        hidden_layers = [n] + [x] + [r]
        hidden_layers_list.append(hidden_layers)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        return x*(1-x)

    def mse(y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))/(2*len(y_true))

    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size

    train_accuracies = []
    test_accuracies = []
    times = []
    for hidden_layers in   (hidden_layers_list):
        start_time = time.time()
        iter = 0
        error = 100
        input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
        output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
        weights = [np.random.randn(output_sizes[i],input_sizes[i])*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        bias = [np.random.randn(output_sizes[i],1)*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        prev_cost = 1e9
        curr_cost = 0
        while (iter < num_iter) :
            prev_cost = curr_cost
            iter += 1
            rate = lr
            rate = lr/iter**0.5
            error = 0
            for k in range(0,m,b):
                x = x_train[k:k+b]
                y_true = y_train[k:k+b]
                
                output = x
                o_ls = [output]
                inputs=[]
                for i in range(len(hidden_layers)-1):
                    #output = np.reshape(output, (1,-1))
                    inputs.append(output)
                    output = np.dot(output, weights[i].T) + bias[i].T
                    output = sigmoid(output)
                    o_ls.append(output)
                
                curr_cost += mse(y_true,output)
                output_error = (y_true-output)*(sigmoid_prime(output))
                deltas = [output_error]
                for i in range(len(hidden_layers)-2,0,-1):
                    deltas.append(np.matmul(deltas[-1], weights[i])*sigmoid_prime(o_ls[i]))

                deltas.reverse()
                
                for i in range(len(hidden_layers)-1):
                    weights[i] += rate*np.matmul(deltas[i].T,o_ls[i])/b
                    bias[i] += rate*np.sum(deltas[i],axis=0,keepdims=True).T/b
            curr_cost/=(m/b)
            
        time_taken = time.time() - start_time
        acc = 0

        for k in range(m):
            output = x_train[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)

            index = np.argmax(output)
            #out += (index)
            #out += (y_true[k])
            acc += (index == y_trueval[k])
        out += ('Accuracy of train is ' + str(acc/m))
        train_accuracies.append(acc/m)

        indices = []
        acc=0.0
        for k in range(len(x_test)):
            output = x_test[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)

            index = np.argmax(output)
            indices.append(index)
            acc += (index == y_test[k])
        out += ('Accuracy of test is ' + str(acc/len(x_test)))
        test_accuracies.append(acc/len(x_test))
        
        y_test = y_test.flatten()
        indices = np.array(indices)
        out += (confusion_matrix(y_test,indices))
        
        times.append(time_taken)
    plt.plot([5,10,15,20,25],train_accuracies)
    plt.xlabel('Hidden layers')
    plt.ylabel('Train accuracies')
    plt.xlabel('Hidden layers')
    plt.ylabel('Test accuracies')
    plt.plot([5,10,15,20,25],test_accuracies)
    plt.xlabel('Hidden layers')
    plt.ylabel('Times taken')
    plt.plot([5,10,15,20,25],times)

    output_file = open(output_path + '/c.txt','w')
    output_file.write(out)

elif q_part == 'd':
     
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import confusion_matrix

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.to_numpy()
    random.shuffle(x_train)
    y_trueval = x_train[:,-1]
    y_train = x_train[:,-1]
    y_train = y_train.reshape((-1,1))
    x_test = test_data.to_numpy()
    y_test = test_data.to_numpy()[:,-1]

    x_train = np.delete(x_train,784,axis=1)
    x_test = np.delete(x_test,784,axis=1)

    x_train = x_train.astype('float64')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.astype('float64')
    x_test /= 255

    hidden_layers_list = []

    lr = 0.1
    num_iter = 1000
    m = len(x_train)
    n = len(x_train[0])
    r = 10
    b = 100

    for x in [100]:
        hidden_layers = [n] + [x] + [x] + [r]
        hidden_layers_list.append(hidden_layers)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        return x*(1-x)

    def mse(y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))/(2*len(y_true))

    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size


    def relu(z):
        return np.maximum(0, z)
    def relu_prime(z):
        return 1. * (z > 0)

    for hidden_layers in   (hidden_layers_list):
        start_time = time.time()
        iter = 0
        error = 100
        input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
        output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
        weights = [np.random.randn(output_sizes[i],input_sizes[i])*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        bias = [np.random.randn(output_sizes[i],1)*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        prev_cost = 1e9
        curr_cost = 0
        while (iter < num_iter) :
            prev_cost = curr_cost
            iter += 1
            rate = lr
            rate = lr/iter**0.5
            error = 0
            for k in range(0,m,b):
                x = x_train[k:k+b]
                y_true = y_train[k:k+b]
                
                output = x
                o_ls = [output]
                inputs=[]
                for i in range(len(hidden_layers)-1):
                    #output = np.reshape(output, (1,-1))
                    inputs.append(output)
                    output = np.dot(output, weights[i].T) + bias[i].T
                    if i == len(hidden_layers)-2:
                        output = sigmoid(output)
                    else:
                        output = relu(output)
                    o_ls.append(output)
                
                curr_cost += mse(y_true,output)
                
                output_error = (y_true-output)*(sigmoid_prime(output))
                deltas = [output_error]
                for i in range(len(hidden_layers)-2,0,-1):
                    deltas.append(np.matmul(deltas[-1], weights[i])*relu_prime(o_ls[i]))

                deltas.reverse()
                
                for i in range(len(hidden_layers)-1):
                    weights[i] += rate*np.matmul(deltas[i].T,o_ls[i])/b
                    bias[i] += rate*np.sum(deltas[i],axis=0,keepdims=True).T/b
            curr_cost/=(m/b)
            
        time_taken = time.time() - start_time
        acc = 0

        for k in range(m):
            output = x_train[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                if i == len(hidden_layers)-2:
                    output = sigmoid(output)
                else:
                    output = relu(output)
            index = np.argmax(output)
            acc += (index == y_trueval[k])
        out += ('Accuracy of train is ' + str(acc/m))

        indices = []
        acc=0.0
        for k in range(len(x_test)):
            output = x_test[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                if i == len(hidden_layers)-2:
                    output = sigmoid(output)
                else:
                    output = relu(output)

            index = np.argmax(output)
            indices.append(index)
            acc += (index == y_test[k])
        out += ('Accuracy of test is ' + str(acc/len(x_test)))
        
        y_test = y_test.flatten()
        indices = np.array(indices)
        out += (confusion_matrix(y_test,indices))

     
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import confusion_matrix

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.to_numpy()
    random.shuffle(x_train)
    y_trueval = x_train[:,-1]
    y_train = x_train[:,-1]
    y_train = y_train.reshape((-1,1))
    x_test = test_data.to_numpy()
    y_test = test_data.to_numpy()[:,-1]

    x_train = np.delete(x_train,784,axis=1)
    x_test = np.delete(x_test,784,axis=1)

    x_train = x_train.astype('float64')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.astype('float64')
    x_test /= 255

    hidden_layers_list = []
    lr = 0.1
    num_iter = 1000
    m = len(x_train)
    n = len(x_train[0])
    r = 10
    b = 100

    for x in [100]:
        hidden_layers = [n] + [x] + [x] + [r]
        hidden_layers_list.append(hidden_layers)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        return x*(1-x)

    def mse(y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))/(2*len(y_true))

    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size


    def relu(z):
        return np.maximum(0, z)
    def relu_prime(z):
        return 1. * (z > 0)

    for hidden_layers in   (hidden_layers_list):
        start_time = time.time()
        iter = 0
        error = 100
        input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
        output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
        weights = [np.random.randn(output_sizes[i],input_sizes[i])*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        bias = [np.random.randn(output_sizes[i],1)*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        prev_cost = 1e9
        curr_cost = 0
        while (iter < num_iter) :
            prev_cost = curr_cost
            iter += 1
            rate = lr
            rate = lr/iter**0.5
            error = 0
            for k in range(0,m,b):
                x = x_train[k:k+b]
                y_true = y_train[k:k+b]
                
                output = x
                o_ls = [output]
                inputs=[]
                for i in range(len(hidden_layers)-1):
                    #output = np.reshape(output, (1,-1))
                    inputs.append(output)
                    output = np.dot(output, weights[i].T) + bias[i].T
                    output = sigmoid(output)
                    o_ls.append(output)
                curr_cost += mse(y_true,output)
                
                output_error = (y_true-output)*(sigmoid_prime(output))
                deltas = [output_error]
                for i in range(len(hidden_layers)-2,0,-1):
                    deltas.append(np.matmul(deltas[-1], weights[i])*sigmoid_prime(o_ls[i]))

                deltas.reverse()
                
                for i in range(len(hidden_layers)-1):
                    weights[i] += rate*np.matmul(deltas[i].T,o_ls[i])/b
                    bias[i] += rate*np.sum(deltas[i],axis=0,keepdims=True).T/b
            curr_cost/=(m/b)
            
        time_taken = time.time() - start_time
        acc = 0

        for k in range(m):
            output = x_train[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)
            index = np.argmax(output)
            #out += (index)
            #out += (y_true[k])
            acc += (index == y_trueval[k])
        out += ('Accuracy of train is ' + str(acc/m))

        indices = []
        acc=0.0
        for k in range(len(x_test)):
            output = x_test[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)

            index = np.argmax(output)
            indices.append(index)
            acc += (index == y_test[k])
        out += ('Accuracy of test is ' + str(acc/len(x_test)))
        y_test = y_test.flatten()
        indices = np.array(indices)
        out += (confusion_matrix(y_test,indices))

    output_file = open(output_path + '/d.txt','w')
    output_file.write(out)

elif q_part == 'e':
     
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import confusion_matrix

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.to_numpy()
    random.shuffle(x_train)
    y_trueval = x_train[:,-1]
    y_train = x_train[:,-1]
    y_train = y_train.reshape((-1,1))
    x_test = test_data.to_numpy()
    y_test = test_data.to_numpy()[:,-1]

    x_train = np.delete(x_train,784,axis=1)
    x_test = np.delete(x_test,784,axis=1)

    x_train = x_train.astype('float64')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.astype('float64')
    x_test /= 255
    hidden_layers_list = []

    lr = 0.1
    num_iter = 1000
    m = len(x_train)
    n = len(x_train[0])
    r = 10
    b = 100

    for x in [2,3,4,5]:
        hidden_layers = [n] + [50]*x + [r]
        hidden_layers_list.append(hidden_layers)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        return x*(1-x)

    def mse(y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))/(2*len(y_true))

    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size


    def relu(z):
        return np.maximum(0, z)

    def relu_prime(z):
        return 1. * (z > 0)

    for hidden_layers in   (hidden_layers_list):
        start_time = time.time()
        iter = 0
        error = 100
        input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
        output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
        weights = [np.random.randn(output_sizes[i],input_sizes[i])*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        bias = [np.random.randn(output_sizes[i],1)*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        prev_cost = 1e9
        curr_cost = 0
        while (iter < num_iter) :
            prev_cost = curr_cost
            iter += 1
            rate = lr
            #rate = lr/iter**0.5
            error = 0
            for k in range(0,m,b):
                x = x_train[k:k+b]
                y_true = y_train[k:k+b]
                
                output = x
                o_ls = [output]
                inputs=[]
                for i in range(len(hidden_layers)-1):
                    #output = np.reshape(output, (1,-1))
                    inputs.append(output)
                    output = np.dot(output, weights[i].T) + bias[i].T
                    if i == len(hidden_layers)-2:
                        output = sigmoid(output)
                    else:
                        output = relu(output)
                    o_ls.append(output)
                
                curr_cost += mse(y_true,output)
                output_error = (y_true-output)*(sigmoid_prime(output))
                deltas = [output_error]
                for i in range(len(hidden_layers)-2,0,-1):
                    deltas.append(np.matmul(deltas[-1], weights[i])*relu_prime(o_ls[i]))

                deltas.reverse()
                
                for i in range(len(hidden_layers)-1):
                    weights[i] += rate*np.matmul(deltas[i].T,o_ls[i])/b
                    bias[i] += rate*np.sum(deltas[i],axis=0,keepdims=True).T/b
            curr_cost/=(m/b)
            
        time_taken = time.time() - start_time
        acc = 0

        for k in range(m):
            output = x_train[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                if i == len(hidden_layers)-2:
                    output = sigmoid(output)
                else:
                    output = relu(output)
            index = np.argmax(output)
            #out += (index)
            #out += (y_true[k])
            acc += (index == y_trueval[k])
        out += ('Accuracy of train is ' + str(acc/m))
        indices = []
        acc=0.0
        for k in range(len(x_test)):
            output = x_test[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                if i == len(hidden_layers)-2:
                    output = sigmoid(output)
                else:
                    output = relu(output)

            index = np.argmax(output)
            indices.append(index)
            acc += (index == y_test[k])
        out += ('Accuracy of test is ' + str(acc/len(x_test)))
     
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import confusion_matrix

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.to_numpy()
    random.shuffle(x_train)
    y_trueval = x_train[:,-1]
    y_train = x_train[:,-1]
    y_train = y_train.reshape((-1,1))
    x_test = test_data.to_numpy()
    y_test = test_data.to_numpy()[:,-1]

    x_train = np.delete(x_train,784,axis=1)
    x_test = np.delete(x_test,784,axis=1)
    x_train = x_train.astype('float64')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.astype('float64')
    x_test /= 255

    hidden_layers_list = []

    lr = 0.1
    num_iter = 1000
    m = len(x_train)
    n = len(x_train[0])
    r = 10
    b = 100

    for x in [2,3,4,5]:
        hidden_layers = [n] + [50]*x + [r]
        hidden_layers_list.append(hidden_layers)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        return x*(1-x)

    def mse(y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))/(2*len(y_true))

    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size
    def relu(z):
        return np.maximum(0, z)

    def relu_prime(z):
        return 1. * (z > 0)
    train_accuracies = []
    test_accuracies = []

    for hidden_layers in   (hidden_layers_list):
        start_time = time.time()
        iter = 0
        error = 100
        input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
        output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
        weights = [np.random.randn(output_sizes[i],input_sizes[i])*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        bias = [np.random.randn(output_sizes[i],1)*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        prev_cost = 1e9
        curr_cost = 0
        while (iter < num_iter) :
            prev_cost = curr_cost
            iter += 1
            rate = lr
            #rate = lr/iter**0.5
            error = 0
            for k in range(0,m,b):
                x = x_train[k:k+b]
                y_true = y_train[k:k+b]
                
                output = x
                o_ls = [output]
                inputs=[]
                for i in range(len(hidden_layers)-1):
                    inputs.append(output)
                    output = np.dot(output, weights[i].T) + bias[i].T
                    output = sigmoid(output)
                    o_ls.append(output)
                
                curr_cost += mse(y_true,output)
                
                output_error = (y_true-output)*(sigmoid_prime(output))
                deltas = [output_error]
                for i in range(len(hidden_layers)-2,0,-1):
                    deltas.append(np.matmul(deltas[-1], weights[i])*sigmoid_prime(o_ls[i]))

                deltas.reverse()
                
                for i in range(len(hidden_layers)-1):
                    weights[i] += rate*np.matmul(deltas[i].T,o_ls[i])/b
                    bias[i] += rate*np.sum(deltas[i],axis=0,keepdims=True).T/b
            curr_cost/=(m/b)
            
        time_taken = time.time() - start_time
        acc = 0

        for k in range(m):
            output = x_train[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)
            index = np.argmax(output)
            #out += (index)
            #out += (y_true[k])
            acc += (index == y_trueval[k])
        out += ('Accuracy of train is ' + str(acc/m))
        train_accuracies.append(acc/m)

        indices = []
        acc=0.0
        for k in range(len(x_test)):
            output = x_test[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)

            index = np.argmax(output)
            indices.append(index)
            acc += (index == y_test[k])
        out += ('Accuracy of test is ' + str(acc/len(x_test)))
        test_accuracies.append(acc/len(x_test))
    plt.plot([2,3,4,5],train_accuracies)
    plt.xlabel('Hidden layers')
    plt.ylabel('Train accuracies')
    plt.plot([2,3,4,5],test_accuracies)
    plt.xlabel('Hidden layers')
    plt.ylabel('Test accuracies')

    output_file = open(output_path + '/e.txt','w')
    output_file.write(out)

elif q_part == 'f':
     
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.metrics import confusion_matrix

    train_path = 'fmnist_train.csv'
    test_path = 'fmnist_test.csv'

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.to_numpy()
    random.shuffle(x_train)
    y_trueval = x_train[:,-1]
    y_train = x_train[:,-1]
    y_train = y_train.reshape((-1,1))
    x_test = test_data.to_numpy()
    y_test = test_data.to_numpy()[:,-1]

    x_train = np.delete(x_train,784,axis=1)
    x_test = np.delete(x_test,784,axis=1)

    x_train = x_train.astype('float64')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.astype('float64')
    x_test /= 255
    hidden_layers_list = []

    lr = 0.1
    num_iter = 1000
    m = len(x_train)
    n = len(x_train[0])
    r = 10
    b = 100

    for x in [3]:
        hidden_layers = [n] + [50]*x + [r]
        hidden_layers_list.append(hidden_layers)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        return x*(1-x)

    def mse(y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))/(2*len(y_true))

    def relu(z):
        return np.maximum(0, z)

    def relu_prime(z):
        return 1. * (z > 0)

    for hidden_layers in   (hidden_layers_list):
        start_time = time.time()
        iter = 0
        error = 100
        input_sizes = [hidden_layers[i] for i in range(len(hidden_layers)-1)]
        output_sizes = [hidden_layers[i] for i in range(1,len(hidden_layers))]
        weights = [np.random.randn(output_sizes[i],input_sizes[i])*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        bias = [np.random.randn(output_sizes[i],1)*(2/input_sizes[i])**0.5 for i in range(len(input_sizes))]
        prev_cost = 1e9
        curr_cost = 0
        while (iter < num_iter) :
            prev_cost = curr_cost
            iter += 1
            rate = lr
            #rate = lr/iter**0.5
            error = 0
            bce_derivs = []
            for k in range(0,m,b):
                x = x_train[k:k+b]
                y_true = y_train[k:k+b]
                
                output = x
                o_ls = [output]
                inputs=[]
                for i in range(len(hidden_layers)-1):
                    #output = np.reshape(output, (1,-1))
                    inputs.append(output)
                    output = np.dot(output, weights[i].T) + bias[i].T
                    output = sigmoid(output)
                    o_ls.append(output)
                
                curr_cost += mse(y_true,output)
                bce_deriv = (y_true/output)-((1-y_true)/(1-output))
                bce_derivs.append(np.sum(bce_deriv,axis=0))
                output_error = (bce_deriv)*(sigmoid_prime(output))
                
                deltas = [output_error]
                for i in range(len(hidden_layers)-2,0,-1):
                    deltas.append(np.matmul(deltas[-1], weights[i])*sigmoid_prime(o_ls[i]))

                deltas.reverse()
                
                for i in range(len(hidden_layers)-1):
                    weights[i] += rate*np.matmul(deltas[i].T,o_ls[i])/b
                    bias[i] += rate*np.sum(deltas[i],axis=0,keepdims=True).T/b
            final_result = np.sum(bce_derivs,axis=0)
            out += (final_result)
            curr_cost/=(m/b)
        time_taken = time.time() - start_time
        acc = 0

        for k in range(m):
            output = x_train[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)
            index = np.argmax(output)
            acc += (index == y_trueval[k])
        out += ('Accuracy of train is ' + str(acc/m))

        indices = []
        acc=0.0
        for k in range(len(x_test)):
            output = x_test[k]
            for i in range(len(hidden_layers)-1):
                output = np.reshape(output, (1,-1))
                output = np.dot(output, weights[i].T) + bias[i].T
                output = sigmoid(output)

            index = np.argmax(output)
            indices.append(index)
            acc += (index == y_test[k])
        out += ('Accuracy of test is ' + str(acc/len(x_test)))

    output_file = open(output_path + '/f.txt','w')
    output_file.write(out)

elif q_part == 'g':
     
    import numpy as np
    import pandas as pd
    import random
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import time
    from sklearn.neural_network import MLPClassifier

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

    clf = MLPClassifier(solver='sgd', max_iter=1000, learning_rate_init=0.1, activation='relu', hidden_layer_sizes=[50,50,50]).fit(x_train, y_train)
    acc_train = clf.score(x_train,y_train)
    out += ("Training accuracy is " + str(acc_train))
    acc_test = clf.score(x_test, y_test)
    out += ("Test accuracy is " + str(acc_test))
    output_file = open(output_path + '/g.txt','w')
    output_file.write(out)

else:
    out += 'wrong qpart'
    output_file = open(output_path + '/w.txt','w')
    output_file.write(out)


output_file.close()