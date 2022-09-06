import q2_b
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import q2_b

def get_stopping_condition (b):
  if b == 1:
    return 0.1
  elif b == 100:
    return 0.9
  elif b == 10000:
    return 0.9
  else:
    return 1.2

def get_1D(lis):
  lis_1D = []
  for l in lis:
    lis_1D.append(l.T[0][0])
  return lis_1D

logs = {}
for r in [1,100,10000,1000000]:
  theta,log = q2_b.sgd(q2_b.arrX,q2_b.arrY,q2_b.m,0.001,r,get_stopping_condition(r))
  logs[r] = log
  plt.plot(get_1D(log["theta"]),log["error"])
  plt.show()
  print(q2_b.find_cost(q2_b.arrX,q2_b.arrY,theta,q2_b.m,0))
  print(q2_b.find_cost(q2_b.testX,q2_b.testY,theta,q2_b.testm,0))


for log in logs.items():
    print(log[0])
    theta = log[1]["theta"]
    print(theta[0].shape)
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