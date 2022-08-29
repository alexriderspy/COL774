import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import q1

def get_3d_meshgrid(theta, margins):
    m0, m1 = margins
    theta = theta.flatten()
    #Adjust margins as per requirement.
    theta0 = np.linspace(theta[0] - m0, theta[0] + m0, 100)
    theta1 = np.linspace(theta[1] - m1, theta[1] + m1, 100)
    X, Y = np.meshgrid(theta0, theta1)
    Z = np.zeros((theta0.size,  theta1.size))

    for i, w0 in enumerate(theta0):
        for j, w1 in enumerate(theta1):
            w = np.expand_dims([w0, w1], axis=0)
            w_ = w.reshape(2,1)
            Z[i, j]= q1.find_cost(q1.arrX, q1.arrY, w_)
    return(X, Y, Z)

def plot_annotated_cost_surface(theta, logs, y_label, margins):
    X, Y, Z = get_3d_meshgrid(theta, margins)
    
    fig = plt.figure(figsize=(12,8))
    graph = plt.subplot(projection='3d')
        
    #Plot the 3D surface
    graph.plot_surface(X, Y, Z, cmap=cm.winter)
    graph.set_xlim(X.min(), X.max())
    graph.set_ylim(Y.min(), Y.max())
    graph.set_zlim3d(top=Z.max())
    graph.set_title('Loss Surface & Learning Curve')
    graph.set_xlabel('theta0')
    graph.set_ylabel('theta1')
    graph.set_zlabel('J(theta)')
            
    #Annotate the surface thetaith learning curves
    for label, log in logs.items():
        theta_array = np.vstack(tuple(log["theta"]))
        print(theta_array)
        if (len(logs) == 1):
            graph.plot(theta_array[:, 0], theta_array[:, 1], log["train_loss"], "ro-", linewidth=3, label=f"{y_label} = {label}")
        else:
            graph.plot(theta_array[:, 0], theta_array[:, 1], log["train_loss"], "o-", linewidth=3, label=f"{y_label} = {label}")
    graph.legend()
    plt.show()

logs = {q1.log["alpha"]: q1.log}
plot_annotated_cost_surface(q1.theta, logs, "alpha", margins=(0.01, 0.001))
