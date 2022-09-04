import matplotlib.pyplot as plt
import numpy as np
import q4_a,q4_c

arrX_Y = np.append(q4_a.arrX,q4_a.arrY,axis=1)

filter = np.asarray([1])

arrX_Y_1 = arrX_Y[np.in1d(arrX_Y[:, 2], filter)]

filter = np.asarray([0])
arrX_Y_0 = arrX_Y[np.in1d(arrX_Y[:, 2], filter)]

plt.plot(arrX_Y_1[:,0].T,arrX_Y_1[:,1].T,'bx')
plt.plot(arrX_Y_0[:,0].T,arrX_Y_0[:,1].T,'rx')

X = q4_c.X
Y = q4_c.Y

plt.plot(X,Y)

plt.show()