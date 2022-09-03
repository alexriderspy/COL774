import matplotlib.pyplot as plt
import q1

arrX = q1.arrX.T[1]
arrY = q1.arrY.T[0]

print(arrX)
print(arrY)

plt.plot(arrX,arrY,'bo')
plt.plot(arrX,q1.Y_pred.T[0])

plt.show()
