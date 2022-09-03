import q2_b
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

logs = q2_b.logs
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
    print(len(theta0),len(theta1),len(theta2))
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(theta0,theta1,theta2,c=theta2,cmap = 'Greens')
    plt.show()