import q1_a
import q1_d
import q1_c

learning_rates = [0.001,0.025,0.1]

for l in learning_rates:
    log,theta = q1_a.linear_regression(l)
    logs = {log["alpha"] : log}
    q1_c.plot_annotated_cost_surface(theta,logs,margins=(0.01,0.001))
    q1_d.plot_annotated_cost_contour(theta, logs, margins=(1, 1))