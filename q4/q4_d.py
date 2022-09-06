import q4
import numpy as np

phi = q4.phi
mu0 = q4.mu0
mu1 = q4.mu1

sigma1 = (1/np.sum(q4.arrY)) * np.matmul((q4.arrY*((q4.arrX.T - mu1).reshape((q4.m,2)))).T,(q4.arrY*((q4.arrX.T - mu1).reshape((q4.m,2))))).reshape((2,2))
sigma0 = (1/np.sum(1-q4.arrY)) * np.matmul(((1-q4.arrY)*((q4.arrX.T - mu0).reshape((q4.m,2)))).T,((1-q4.arrY)*((q4.arrX.T - mu0).reshape((q4.m,2))))).reshape((2,2))

inv_sigma0 = np.linalg.pinv(sigma0)
inv_sigma1 = np.linalg.pinv(sigma1)

print("generalised")
print(sigma1)
print(sigma0)