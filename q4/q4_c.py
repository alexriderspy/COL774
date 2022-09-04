import q4_a
import numpy as np

phi = q4_a.phi
mu0 = q4_a.mu0
mu1 = q4_a.mu1
sigma = q4_a.sigma

inv_sigma = np.linalg.pinv(sigma)

a=inv_sigma[0][0]
b=inv_sigma[0][1]
c=inv_sigma[1][0]
d=inv_sigma[1][1]

mu01 = mu0[0][0]
mu02 = mu0[1][0]

mu11 = mu1[0][0]
mu12 = mu1[1][0]

intercept = 2*np.log((1-phi)/phi) + a*(mu01 + mu11)*(mu11-mu01) + d*(mu02+mu12)*(mu12-mu02) - (b+c)*(mu01*mu02 - mu11*mu12)

X1_min = -0.5
X1_max = 0.5
X = np.linspace(X1_min,X1_max,10)
Y = (intercept - X*(2*a - mu02*(b+c) + mu12*(b+c)))/(2*d - mu01*(b+c) + mu11*(b+c))
