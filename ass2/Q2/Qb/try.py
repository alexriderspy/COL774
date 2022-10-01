import numpy as np

l = np.array([5.9,2]).reshape(2,1).astype('uint8')
X = np.array([[10,11,23,12,45,67],[10,11,9,10,42,67]])

array_images_top5 = np.append(l,X,axis=1)
array_images_top5.sort()
array_images_top5 = array_images_top5[-5:]
array_images_top5 = np.delete(array_images_top5,0,1)

print(array_images_top5)