import q4_a
import numpy as np

phi = q4_a.phi
mu0 = q4_a.mu0
mu1 = q4_a.mu1

sigma1 = (1/np.sum(q4_a.arrY)) * np.matmul((q4_a.arrY*((q4_a.arrX.T - mu1).reshape((q4_a.m,2)))).T,(q4_a.arrY*((q4_a.arrX.T - mu1).reshape((q4_a.m,2))))).reshape((2,2))
sigma0 = (1/np.sum(1-q4_a.arrY)) * np.matmul(((1-q4_a.arrY)*((q4_a.arrX.T - mu0).reshape((q4_a.m,2)))).T,((1-q4_a.arrY)*((q4_a.arrX.T - mu0).reshape((q4_a.m,2))))).reshape((2,2))

inv_sigma0 = np.linalg.pinv(sigma0)
inv_sigma1 = np.linalg.pinv(sigma1)

print(inv_sigma0)
print(inv_sigma1)