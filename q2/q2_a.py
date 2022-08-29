import numpy as np
import math
#sampling

def n_34(x):
    std = 2
    mean = 3
    return (1/(std * math.sqrt(2*math.pi)))*(math.exp((-0.5*(((x-mean)/std)**2))))

def n_14(x):
    std = 2
    mean = -1
    return (1/(std * math.sqrt(2*math.pi)))*(math.exp((-0.5*(((x-mean)/std)**2))))

def n_02(x):
    std = math.sqrt(2)
    mean = 0
    return (1/(std * math.sqrt(2*math.pi)))*(math.exp((-0.5*(((x-mean)/std)**2))))

m = 1000000
X = np.linspace(-100,100,num=m)

arrX = np.random.choice(X, size = m, replace = False)

normal_vec = np.vectorize(n_34)
arrX1 = normal_vec(arrX)

normal_vec = np.vectorize(n_14)
arrX2 = normal_vec(arrX)

arrX0 = np.ones(m)
normal_vec = np.vectorize(n_02)
arre = normal_vec(arrX)

theta = np.array([3,1,2])

theta.reshape((3,1))

arrX = arrX0.reshape((m,1))
arrX = np.append(arrX,arrX1.reshape((m,1)),axis=1)
arrX = np.append(arrX,arrX2.reshape((m,1)),axis=1)

arrY = np.matmul(arrX,theta).reshape((m,1))