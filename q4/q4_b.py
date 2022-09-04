import matplotlib.pyplot as plt
import numpy as np
import q4_a

arrX_Y = np.append(q4_a.arrX,q4_a.arrY,axis=1)

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

plt.show()