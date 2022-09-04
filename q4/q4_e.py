import q4
import numpy as np

from sympy import plot_implicit, symbols, Eq, And
x1, x2 = symbols('x y')

phi = q4.phi
mu0 = q4.mu0
mu1 = q4.mu1

inv_sigma0 = q4.inv_sigma0
inv_sigma1 = q4.inv_sigma1

mu01 = mu0[0][0]
mu02 = mu0[1][0]

mu11 = mu1[0][0]
mu12 = mu1[1][0]

a1=inv_sigma0[0][0]
b1=inv_sigma0[0][1]
c1=inv_sigma0[1][0]
d1=inv_sigma0[1][1]

a2=inv_sigma1[0][0]
b2=inv_sigma1[0][1]
c2=inv_sigma1[1][0]
d2=inv_sigma1[1][1]

plot_implicit(Eq(a1*(x1*x1+mu01*mu01-2*x1*mu01) + (b1+c1)*(x2*x1 - mu02*x1-mu01*x2+mu01*mu02) + d1*(x2*x2 + mu02*mu02-2*x2*mu02),(2*np.log((1-phi)/phi))+a2*(x1*x1+mu11*mu11-2*x1*mu11) + (b2+c2)*(x1*x2-x1*mu12-x2*mu11+mu11*mu12) + d2*(x2*x2+mu12*mu12-2*x2*mu12)),(x1,0,100))