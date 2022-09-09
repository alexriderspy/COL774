import matplotlib.pyplot as plt
import q1_a

arrX = q1_a.arrX.T[1]
arrY = q1_a.arrY.T[0]

plt.plot(arrX,arrY,'bo')
plt.plot(arrX,q1_a.Y_pred.T[0])

plt.show()
