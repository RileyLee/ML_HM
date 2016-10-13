import numpy as np
from Lasso import lasso

# Load a text file of integers:
y = np.loadtxt("../hw1-data/upvote_labels.txt", dtype=np.int)

# Load a text file of strings:
featureNames = open("../hw1-data/upvote_features.txt").read().splitlines()

# Load a csv of floats:
A = np.genfromtxt("../hw1-data/upvote_data.csv", delimiter=",")



model = lasso(10.0, 0.1, 0.99);

model.mountData(A[range(0,4000,1), :], y[range(0, 4000, 1)]);
model.mountName(featureNames)
model.fit();

print("Training Accuracy:")
[Err, sqr_Err, RMSE] = model.predict();

model.mountData(A[range(4000,5000,1), :], y[range(4000, 5000, 1)]);
print("Validation Accuracy:")
[Err, sqr_Err, RMSE] = model.predict();