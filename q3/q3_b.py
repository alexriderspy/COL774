import matplotlib.pyplot as plt
import numpy as np
import q3

arrX_Y = np.append(q3.arrX,q3.arrY,axis=1)

X1_min = np.min(arrX_Y[:,1])
X1_max = np.max(arrX_Y[:,1])

filter = np.asarray([1])

arrX_Y_1 = arrX_Y[np.in1d(arrX_Y[:, 3], filter)]

filter = np.asarray([0])
arrX_Y_0 = arrX_Y[np.in1d(arrX_Y[:, 3], filter)]

print(arrX_Y_1[:,1].T)
print(arrX_Y_1[:,2].T)
print(arrX_Y_0[:,1].T)
print(arrX_Y_0[:,2].T)

plt.plot(arrX_Y_1[:,1].T,arrX_Y_1[:,2].T,'bx')
plt.plot(arrX_Y_0[:,1].T,arrX_Y_0[:,2].T,'rx')

X = np.linspace(X1_min,X1_max,100)
theta = q3.theta

Y = (-theta[0][0]-theta[1][0]*X)/theta[2][0]

plt.plot(X,Y)
plt.show()
