import numpy as np

m = 1000000
mu, sigma = 3,2
x0 = np.random.normal(mu, sigma, m)
mu, sigma = -1,2
x1 = np.random.normal(mu, sigma, m)
mu, sigma = 0,np.sqrt(2)
e = np.random.normal(mu, sigma, m)

arrX = np.ones(m).reshape((m,1))

arrX = np.append(arrX,x0.reshape((m,1)),axis=1)
arrX = np.append(arrX,x1.reshape((m,1)),axis=1)

theta = np.array([3, 1, 2]).reshape((3,1))
arrY = np.matmul(arrX, theta) + e.reshape((m,1))
print(arrY)