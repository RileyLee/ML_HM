import numpy as np
import matplotlib.pyplot
import mnist
import scipy
import scipy.sparse.linalg
import scipy.misc
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.spatial import distance
from numpy import linalg as LA
from collections import Counter
import random
import pdb

trainX, trainY = mnist.load_mnist("training", None, './MNIST');
testX, testY = mnist.load_mnist("testing", None, './MNIST');
print("MNIST loaded")

class KMeans:    
    
    def __init__(self, k, thresh):
        self.sigma = 0;
        self.K = k;
        self.thresh = thresh
        
    def load_train(self, trainX, trainY):
        s = trainX.shape;
        self.n = s[1] * s[2];
        #self.n = s[1]
        self.N = s[0];
        self.oriX = np.reshape(trainX, (self.N, self.n));
        self.X = np.asmatrix(self.oriX);
        self.d = self.n;
        self.Y = trainY
        print("Training data loaded...")
    
    def load_test(self, testX, testY):
        self.testN = testX.shape[0];
        self.testX = np.asmatrix(np.reshape(testX, (self.testN, self.n)));
        self.testY = testY
        print("Testing data loaded...")
    
    def compPCA(self):
        self.Sigma = self.oriX.transpose().dot(self.oriX);
        self.eigW, self.eigV = LA.eig(self.Sigma);
        print("PCA Projection Computed...")
    
    def projPCA(self, d):
        self.d = d;
        self.X = np.asmatrix(self.oriX) * np.asmatrix(self.eigV[:, 0:self.d]);
        print("Convert from " + str(self.n) + ' dimensions to ' + str(d))
    
    def sortToClass(self):
        self.DistMat = distance.cdist(self.X, self.means, 'sqeuclidean');
        self.sortedK = np.argmin(self.DistMat, axis = 1);
        self.DistMin = np.min(self.DistMat, axis = 1);
        
    def UpdateKmeans(self):
        #pdb.set_trace()
        self.kloss = np.mean(self.DistMin);
        for i in range(0,self.K):
            temp = np.asmatrix(np.mean(self.X[self.sortedK==i, :], axis=0))
            if not np.sum(np.isnan(temp))>0:
                self.means[i, :] = temp;
            temp = self.Y[self.sortedK==i]
            if temp.shape[0]>0:
                self.classVal[i] = Counter(temp).most_common(1)[0][0]
        self.iter = self.iter + 1
        self.iterLoss.append(self.kloss);
    
    def printStatus(self):
        print("Processing Iteration " + str(self.iter));
        print("\t Total sum : " + str(self.iterLoss[self.iter-1]))
        
    def initKMeans(self):
        self.iter = 0;
        self.iterLoss = [];
        self.kloss = np.zeros(self.K);
        self.classVal = np.zeros(self.K);
        self.means = np.asmatrix(np.zeros((self.K, self.d)));
        for i in range(0, self.K):
            idx = np.int(np.floor(random.uniform(0, 1) * self.N));
            self.means[i, :] = self.X[idx, :];
        t = np.linspace(0, 2 * np.pi, 20)
        self.colorSet = np.cos(t)
        print("Kmeans of " + str(self.K) + " classes initiated....")
        
    def plotPts(self, percent, axis1, axis2):
        n = np.int(1 / percent)
        X = np.concatenate((self.X[range(0, self.N-1, n), axis1], self.X[range(0, self.N-1, n), axis2]), axis=1)
        color = self.colorSet[self.sortedK[range(0, self.N-1, n)]]
        plot = matplotlib.pyplot.scatter(X[:,0],X[:,1],c=color)
        #matplotlib.pyplot.show()
        matplotlib.pyplot.savefig('foo.png', transparent = True)
        
    def plotMeans(self):
        self.meansplot = np.zeros((28*2, 28*8), dtype=np.double);
        noa = np.zeros((16,2));
        for i in range(0,16):
            noa[i,0] = np.sum(model.sortedK==i);
            noa[i,1] = i;
        
        noa = noa[noa[:,0].argsort()]
        
        for x in range(0,2):
            for y in range(0,8):
                self.meansplot[x*28: (x*28+28), (y*28):(y*28+28)] = np.reshape(self.means[noa[np.int(15-(x*2+y)),1],:], (28, 28));
        scipy.misc.imsave('Means.jpg', np.uint8(self.meansplot * 255));
        
    def classify(self):
        self.DistMatT = distance.cdist(self.testX, self.means, 'sqeuclidean');
        self.classifiedTe = self.classVal[np.argmin(self.DistMatT, axis = 1)];
        self.classifiedTr = self.classVal[np.argmin(self.DistMat, axis = 1)];
        self.loss01Test = 1 - np.float(np.sum(self.classifiedTe==self.testY)) / np.float(self.testN);
        self.loss01Train = 1 - np.float(np.sum(self.classifiedTr==self.Y)) / np.float(self.N);
        print("Training 0/1 loss : " + str(self.loss01Train))
        print("Testing 0/1 loss : " + str(self.loss01Test))
        
        
model = KMeans(250, 0.01);
model.load_train(trainX, trainY);
#model.compPCA();
#model.projPCA(50);
model.initKMeans();
#model.plotMeans()
while model.iter < 5 or model.iterLoss[model.iter-2] - model.iterLoss[model.iter-1] > model.thresh:
    model.sortToClass()
    #model.plotPts(0.1, 0, 1)
    model.UpdateKmeans()
    model.printStatus();
model.plotMeans()

model.load_test(testX, testY);
model.classify();

red_star = matplotlib.pyplot.plot(range(0,model.iter), model.iterLoss, color="blue", hold = True, linewidth=2.0)
matplotlib.pyplot.savefig('Squared Reconstruction Error', transparent = True)
model.plotMeans()

noa = np.zeros(16);
for i in range(0,16):
    noa[i] = np.sum(model.sortedK==i);

noa = np.sort(noa)
noa = noa[::-1]
matplotlib.pyplot.cla()
noa = matplotlib.pyplot.plot(range(1,17), noa, color="blue", hold = True, linewidth=2.0)
matplotlib.pyplot.savefig('Number of Assignments', transparent = True)
matplotlib.pyplot.title('Number of Assignments')
model.plotMeans()