import numpy as np
import matplotlib.pyplot
import mnist
import scipy
import scipy.sparse.linalg
import scipy.misc
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from numpy import linalg as LA

class PCA:
    
    def __init__(self):
        self.sigma = 0;
        
        
    def load_train(self, trainX):
        s = trainX.shape;
        self.d = s[1] * s[2] + 1;
        self.n = self.d;
        self.N = s[0];
        self.X = np.reshape(trainX, (self.N, self.d-1));
    
        self.Y = np.zeros((self.N, 10),dtype=np.float)
        print("Training data loaded...")
        
    def compSigma(self):
        self.Sigma = self.X.transpose().dot(self.X);
        self.w, self.v = LA.eig(self.Sigma);
        print("Sigma Computed...")
    
    def decompose(self, inNumber, k):
        weights = np.zeros(k);
        outImage = np.zeros(inNumber.shape);
        for i in range(0, k):
            weights[i] = inNumber.transpose().dot(np.reshape(self.v[:,i], (784,1)))
            outImage += weights[i] * np.reshape(self.v[:,i], (784,1))
        return outImage