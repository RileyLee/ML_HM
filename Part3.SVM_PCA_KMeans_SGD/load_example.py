import numpy as np
import scipy.io as io
import scipy.sparse as sparse

# Load a text file of integers:
y = np.loadtxt("../hw1-data/upvote_labels.txt", dtype=np.int)

# Load a text file of strings:
featureNames = open("../hw1-data/upvote_features.txt").read().splitlines()

# Load a csv of floats:
A = np.genfromtxt("../hw1-data/upvote_data.csv", delimiter=",")

# Load a matrix market matrix, convert it to csc format:
B = io.mmread("../hw1-data/star_data.mtx").tocsc()

print(A.shape)