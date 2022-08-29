import q1_c
import q1
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_annotated_cost_contour(W, logs, y_label, margins):
    X, Y, Z = q1_c.get_3d_meshgrid(W, margins)
    
    fig = plt.figure(figsize=(12,8))
    graph = plt.subplot()
    
    #Plot the 3D surface
    graph.contourf(X, Y, Z, cmap=cm.winter)
    graph.set_xlim(X.min(), X.max())
    graph.set_ylim(Y.min(), Y.max())
    graph.set_title('Loss Contour & Learning Curve')
    graph.set_xlabel('W0')
    graph.set_ylabel('W1')
        
    #Annotate the surface with learning curves
    for label, log in logs.items():
        W_array = np.vstack(tuple(log["theta"]))
        if (len(logs) == 1):
            graph.plot(W_array[:, 0], W_array[:, 1], "ro-", linewidth=3, label=f"{y_label} = {label}")
        else:
            graph.plot(W_array[:, 0], W_array[:, 1], "o-", linewidth=3, label=f"{y_label} = {label}")
    graph.legend()
    plt.show()

plot_annotated_cost_contour(q1.theta, q1_c.logs, "alpha", margins=(0.01, 10))