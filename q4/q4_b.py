import matplotlib.pyplot as plt
import numpy as np
import q4,q4_e

arrX_Y = np.append(q4.arrX,q4.arrY,axis=1)

filter = np.asarray([1])

arrX_Y_1 = arrX_Y[np.in1d(arrX_Y[:, 2], filter)]

filter = np.asarray([0])
arrX_Y_0 = arrX_Y[np.in1d(arrX_Y[:, 2], filter)]

plt.plot(arrX_Y_1[:,0].T,arrX_Y_1[:,1].T,'bx')
plt.plot(arrX_Y_0[:,0].T,arrX_Y_0[:,1].T,'rx')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(["Canada","Alaska"])

X = q4.X
Y = q4.Y

plt.plot(X,Y)

levels = np.array([0])
cs = plt.contour(q4_e.x1,q4_e.x2,q4_e.z,levels)
plt.clabel(cs,inline=1)
plt.grid(True)
plt.show()