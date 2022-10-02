import numpy as np

l = np.array([5.9,2]).reshape(2,1).astype('uint8')
X = np.array([[10,11,23,12,45,67],[10,11,9,10,42,67]])

array_images_top5 = np.append(l,X,axis=1)
array_images_top5.sort()
array_images_top5 = array_images_top5[-5:]
array_images_top5 = np.delete(array_images_top5,0,1)

print(array_images_top5)

s = np.array([8,9,2,3,4,9,1,22,6,3])
p = np.bitwise_and(s>5,s<20)
ind = np.arange(len(s))[p]
print(ind)
print(p)

table_a = np.zeros((6,6))
table_b = np.zeros((6,6))

for i in range(4):
    table_b[i+1][i+1] += 1

table_b = table_b.astype('str')
for i in range(1,6):
    for j in range(1,6):
        table_b[i][j] = table_b[i][j].tostring()

print(table_b)
table_b[0][1] = '0(A)'
table_b[0][2] = '1(A)'
table_b[0][3] = '2(A)'
table_b[0][4] = '3(A)'
table_b[0][5] = '4(A)'

table_b[1][0] = '0(P)'
table_b[2][0] = '1(P)'
table_b[3][0] = '2(P)'
table_b[4][0] = '3(P)'
table_b[5][0] = '4(P)' 

print(table_b)
print(abs(-12))