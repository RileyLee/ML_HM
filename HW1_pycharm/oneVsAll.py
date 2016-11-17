import numpy as np
import matplotlib
import mnist
import scipy
import scipy.sparse.linalg
from scipy.sparse import csr_matrix
from scipy.sparse import identity
import pdb


trainX, trainY = mnist.load_mnist("training", None, './MNIST');
testX, testY = mnist.load_mnist("testing", None, './MNIST');
print("MNIST loaded")

class oneVsAll:
    def __init__(self, lamda):
        self.lamda = lamda;
        self.weights = {};
        
    def load_data(self, string, ori_X, ori_Y):
        self.trainY = ori_Y;
        s = ori_X.shape;
        self.d = s[1] * s[2] + 1;
        self.n = self.d;
        self.N = s[0];
        self.X = np.reshape(ori_X, (self.N, self.d-1));
        self.X = np.insert(self.X, 0, 1, axis=1);
        self.Y = np.reshape(self.trainY, (self.N, 1))
        print("Loading data " + string + " complete....")

    def makeY(self, val):
        temp = np.array(self.Y == val, dtype=float);
        self.Y = np.reshape(temp, (self.N, 1))
        print("Classifying class " + str(self.digit))
    
    def compWeightBatch(self):
        idx_from = range(0, self.N, self.batchSize)
        idx_to = idx_from[1:]
        idx_to.extend([self.N])
        self.w = np.zeros((self.n, 1), dtype=np.float);
        for x in range(0, len(idx_from)):
            sigma = np.dot(self.X[idx_from[x]:idx_to[x]].transpose(), self.X[idx_from[x]:idx_to[x]]);
            self.A = np.linalg.inv(sigma + self.lamda * np.identity(self.n));
            self.Ax = np.dot(self.A, self.X.transpose());
            self.w += np.dot(self.Ax, self.Y);
        self.weights[self.digit] = self.w;
        
    def compWeightA(self):
        #pdb.set_trace()
        sigma = self.X.transpose() * self.X;
        self.A = np.linalg.inv(sigma + self.lamda * np.identity(self.n));
        self.Ax = np.dot(self.A, self.X.transpose());
        self.w = np.dot(self.Ax, self.Y);
        self.weights[self.digit] = self.w;
        
    def compWeight(self):
        #pdb.set_trace()
        self.X = np.asmatrix(self.X);
        sigma = self.X.transpose() * self.X;
        Ident = np.identity(self.n);
        Ident = np.asmatrix(Ident);
        #pdb.set_trace()
        self.A = np.linalg.inv(sigma + self.lamda * Ident);
        
        self.Ax = self.A * self.X.transpose();
        self.w = self.Ax * self.Y;
        self.weights[self.digit] = self.w;
    
    def multifit(self, batchSize):
        self.batchSize = batchSize;
        for self.digit in range(0,10):
            model.makeY(self.digit);
            model.compWeight();
            
    def predict(self):
        self.predicted = np.zeros((self.N, 10), dtype=float);
        for self.digit in range(0, 10):
            weight = self.weights[self.digit];
            self.predicted[:,self.digit] = np.reshape(np.dot(self.X, weight), self.N);
        self.predLabel = np.argmax(self.predicted, axis=1);
        
    def eval01(self):
        temp = (np.reshape(self.predLabel, (len(self.Y),1)) != np.reshape(self.Y, (len(self.Y),1))).astype(float)
        self.loss01 = np.sum(temp) / len(self.Y);
        print("0 / 1 loss : " + str(self.loss01));
    
    def squareLoss(self):
        
        lttemp = np.reshape(np.float64(self.Y==0),(self.N, 1))
        for i in range(1,10):
            lttemp = np.concatenate([lttemp, np.reshape(np.float64(self.Y==i),(self.N, 1))], axis=1);
        temp = lttemp - self.predicted
        self.sqsum = np.sum(np.square(temp))/self.N
        print("Square Loss : " + str(self.sqsum))
    
    def genGaussFeatWeight(self, n):
        self.n = n;
        self.weight = np.random.randn(self.d, self.n);
    
    def randGaussFeatConv(self):
        self.X = np.dot(self.X, self.weight);
        
model = oneVsAll(1.0);
model.load_data("training", trainX, trainY);
model.multifit(500)
model.predict()

print("Training Dataset : ")
model.Y = trainY
model.eval01()
model.squareLoss()

print("Testing Dataset : ")
model.Y = testY
model.load_data("testing", testX, testY);
model.predict()
model.eval01()
model.squareLoss()