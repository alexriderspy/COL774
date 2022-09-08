import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import q1_a

def get_3d_meshgrid(theta, margins):
    m0, m1 = margins

    theta0 = np.linspace(theta.T[0][0]- m0, theta.T[0][0] + m0, 100)
    theta1 = np.linspace(theta.T[0][1] - m1, theta.T[0][1] + m1, 100)
    X, Y = np.meshgrid(theta0, theta1)
    Z = np.zeros((theta0.size,theta1.size))
    for i,w0 in enumerate(theta0):
        for j,w1 in enumerate(theta1):
            W = np.array([w0,w1]).reshape((2,1))
            Z[i,j] = q1_a.find_cost(q1_a.arrX,q1_a.arrY,W)
    return(X, Y, Z)

def plot_annotated_cost_surface(theta, logs, margins):
    X, Y, Z = get_3d_meshgrid(theta, margins)
    
    fig = plt.figure(figsize=(12,8))
    graph = plt.subplot(projection='3d')

    graph.plot_surface(X, Y, Z, cmap=cm.winter)
    graph.set_xlim(X.min(), X.max())
    graph.set_ylim(Y.min(), Y.max())
    graph.set_zlim3d(top=Z.max())
    graph.set_title('Loss Surface & Learning Curve')
    graph.set_xlabel('theta0')
    graph.set_ylabel('theta1')
    graph.set_zlabel('J(theta)')
            
    for _, log in logs.items():
        theta_array = np.vstack(tuple(log["theta"]))
        graph.scatter(theta_array[:, 0], theta_array[:, 1], log["train_loss"])
    plt.show()

logs = {q1_a.log["alpha"]: q1_a.log}
plot_annotated_cost_surface(q1_a.theta, logs, margins=(0.01, 0.001))
