import numpy as np
from Lasso import lasso
import matplotlib.pyplot

# Load a text file of integers:
y = np.loadtxt("../hw1-data/upvote_labels.txt", dtype=np.int)

# Load a text file of strings:
featureNames = open("../hw1-data/upvote_features.txt").read().splitlines()

# Load a csv of floats:
A = np.genfromtxt("../hw1-data/upvote_data.csv", delimiter=",")

model = lasso(10.0, 0.1, 0.99);

model.mountData(A[range(0,4000,1), :], y[range(0, 4000, 1)], "training");
model.mountData(A[range(4000,5000,1), :], y[range(4000, 5000, 1)], "validation");
model.mountData(A[range(5000,6000,1), :], y[range(5000, 6000, 1)], "testing");

model.fit(True, 2);

red_star = matplotlib.pyplot.plot(model.lamdavals[1:model.iter-1], model.RMSE[1:model.iter-1], color="red", hold = True, markersize=4, label='RMSE training', linewidth=2.0)
blue_star = matplotlib.pyplot.plot(model.lamdavals[1:model.iter-1], model.RMSE_val[1:model.iter-1], hold = True, color="blue", markersize=4, label='RMSE validation', linewidth=2.0)
matplotlib.pyplot.ylim([1.5,2.5])
matplotlib.pyplot.savefig("RMSE.png", transparent = True)


red_star = matplotlib.pyplot.plot(model.lamdavals[1:model.iter-1], model.nonzeros[1:model.iter-1], color="red", hold = False, markersize=4, label='recall sigma=1.0', linewidth=2.0)
matplotlib.pyplot.savefig("nonzeros.png", transparent = True)


model.evalRMSE_te(5);

idx = (np.abs(model.lamdavals - 3.0)).argmin()
w = np.reshape(model.w_history[idx, :], (model.n,1))
print("lambda value : " + str(model.lamda))
print("Testing Accuracy:" + str(model.RMSE_te))

print("Best features : ")
array = np.concatenate([w, np.reshape(np.arange(0, model.n), (model.n, 1))], axis=1)
sorted = array[np.argsort(-array[:,0])]
for i in range(0,10):
    print(featureNames[np.int(sorted[i,1])] + " : " + str(model.w[np.int(sorted[i,1])]))




