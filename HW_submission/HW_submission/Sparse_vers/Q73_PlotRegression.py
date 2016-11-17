from Lasso import lasso
import numpy as np
import matplotlib.pyplot

num_pts = 80;

recall = np.zeros(num_pts);
prec = np.zeros(num_pts);
X = np.zeros(num_pts);

idx = 0;
lamda = 150

print ("Running Simulatio  with lamda " + str(lamda))
model = lasso(lamda, 0.000001, 0.99)
model.generate_data(50, 75, 5, 1.0)
model.fit(False, 1);

red_star = matplotlib.pyplot.plot(model.lamdavals, model.recall, color="red", hold = True, marker="*", markersize=4, label='recall sigma=1.0', linewidth=2.0)
blue_star = matplotlib.pyplot.plot(model.lamdavals, model.precision, hold = True, color="blue", marker="*", markersize=4, label='precision sigma=1.0', linewidth=2.0)


model = lasso(lamda, 0.000001, 0.99)
model.generate_data(50, 75, 5, 10.0)
model.fit(False, 1);

red_circle = matplotlib.pyplot.plot(model.lamdavals, model.recall, color="red", hold = True, marker="o", markersize=4, label='recall sigma=10', linewidth=2.0)
blue_circle = matplotlib.pyplot.plot(model.lamdavals, model.precision, hold = True, color="blue", marker="o", markersize=4, label='precision sigma=10', linewidth=2.0)


matplotlib.pyplot.ylim([0,1.1])
matplotlib.pyplot.savefig("RP_plot.png", transparent = True)
