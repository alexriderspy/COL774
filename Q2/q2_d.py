import q2_b
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import q2_b

def get_stopping_condition (b):
  if b == 1:
    return 0.000000001,1000000
  elif b == 100:
    return 0.01,100000
  elif b == 10000:
    return 0.1,50000
  else:
    return 0.1,50000

def get_1D(lis):
  lis_1D = []
  for l in lis:
    lis_1D.append(l.T[0][0])
  return lis_1D

logs = {}
for r in [1,100,10000,1000000]:
  tuple = get_stopping_condition(r)
  theta,log = q2_b.sgd(q2_b.arrX,q2_b.arrY,q2_b.m,0.001,r,tuple[0],tuple[1])
  logs[r] = log
  plt.plot(get_1D(log["theta"]),log["error"])
  plt.show()


for log in logs.items():
    print(log[0])
    theta = log[1]["theta"]
    theta0 = []
    theta1 = []
    theta2 = []
    for w in theta:
        theta0.append(w[0][0])
        theta1.append(w[1][0])
        theta2.append(w[2][0])
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(theta0,theta1,theta2,c=theta2,cmap = 'Greens')
    plt.show()