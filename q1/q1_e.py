import q1
import q1_d

learning_rates = [0.001,0.025,0.1]

for l in learning_rates:
    theta = q1.linear_regression(l)
    logs = {q1.log["alpha"] : q1.log}
    q1_d.plot_annotated_cost_contour(theta, logs, "alpha", margins=(0.01, 10))