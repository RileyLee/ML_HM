import numpy as np
import matplotlib.pyplot
import mnist
import scipy
import scipy.sparse.linalg
import scipy.misc
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from numpy import linalg as LA
import pdb

trainX, trainY = mnist.load_mnist("training", None, './MNIST');
testX, testY = mnist.load_mnist("testing", None, './MNIST');
print("MNIST loaded")

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
        
model = PCA();
model.load_train(trainX);
model.compSigma();

print("Eigen value 1 : " + str(abs(model.w[0])))
print("Eigen value 2 : " + str(abs(model.w[1])))
print("Eigen value 10 : " + str(abs(model.w[9])))
print("Eigen value 30 : " + str(abs(model.w[29])))
print("Eigen value 50 : " + str(abs(model.w[49])))
print("Sum of eigen values : " + str(np.sum(np.abs(model.w))))
print("Sum of 2 norm : " + str(np.sum(np.square(model.X))))

denom = np.sum(abs(model.w));
Y = np.zeros(49);
X = range(2,51);
for i in range(2,51):
    Y[i-2] = 1 - np.sum(abs(model.w[0:i])) / denom;
    
red_star = matplotlib.pyplot.plot(X, Y, color="blue", hold = True, linewidth=2.0)
matplotlib.pyplot.savefig('Fractional Reconstruction Error1', transparent = True)

# Display the first 10 eigenvectors as images
Image = np.zeros((trainX.shape[1], trainX.shape[2]*10));
for i in range(0,10):
    Image[:, (trainX.shape[2]*i) : (trainX.shape[2]*(i+1))] = np.reshape(model.v[:,i], (trainX.shape[1], trainX.shape[2]))
scipy.misc.imsave('Eigen/Eigen.jpg', np.uint8(Image*255 + 127))


ImageSyn = np.zeros((140, 140), dtype=np.uint8)
Image = np.zeros((28, 140), dtype=np.uint8)
idx = 0;
for im in [100, 101, 102, 103, 104]:
    data = np.reshape(testX[im,:]*255, (784, 1))
    idy = 0;
    for i in [2, 5, 10, 20, 50]:
        outImage = model.decompose(data, i);
        outImage = np.reshape(outImage, (28, 28))
        
        ImageSyn[(idx*28):(idx*28+28), (idy*28):(idy*28+28)] = np.uint8(np.clip(outImage, 0, 255))
        Image[:, (idx*28):(idx*28+28)] = np.uint8(testX[im,:]*255);
        
        idy += 1;
    idx += 1;
scipy.misc.imsave('SynImage.jpg', ImageSyn)
scipy.misc.imsave('Image.jpg', Image)