import q1_c
import q1
import matplotlib.pyplot as plt
import numpy as np

def plot_annotated_cost_contour(W, logs, margins):
    X, Y, Z = q1_c.get_3d_meshgrid(W, margins)
    
    #Plot the 3D surface
    fig = plt.figure(figsize=(12,8))
    graph = plt.subplot()
        
    #Plot the 3D surface
    graph.contourf(X, Y, Z)
    graph.set_xlim(X.min(), X.max())
    graph.set_ylim(Y.min(), Y.max())
    graph.set_title('Loss Surface & Learning Curve')
    graph.set_xlabel('theta0')
    graph.set_ylabel('theta1')
            
    for label, log in logs.items():
        theta_array = np.vstack(tuple(log["theta"]))
        print(theta_array)
        graph.scatter(theta_array[:, 0], theta_array[:, 1])
    graph.legend()
    plt.show()
        
plot_annotated_cost_contour(q1.theta, q1_c.logs, margins=(1, 1))