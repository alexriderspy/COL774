import matplotlib.pyplot as plt
import q1

arrX = q1.arrX.T[0]
arrY = q1.arrY.T[0]
plt.plot(arrX,arrY,'bo')
plt.plot(arrX,q1.Y_pred.T[0])

plt.show()
