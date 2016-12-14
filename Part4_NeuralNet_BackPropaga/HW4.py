import numpy as np
import sys
import matplotlib.pyplot
import mnist
import scipy
import scipy.sparse.linalg
import scipy.misc
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from numpy import linalg as LA
import cv2
import pdb

trainX, trainY = mnist.load_mnist("training", None, './MNIST');
testX, testY = mnist.load_mnist("testing", None, './MNIST');
testX1 = testX

print("MNIST loaded")



class PCA:
    
    def __init__(self):
        self.sigma = 0;
        
    def load_train(self, trainX):
        s = trainX.shape;
        self.d = s[1] * s[2];
        self.N = s[0];
        self.X = np.reshape(trainX, (self.N, self.d));
        print("PCA : Training data loaded...")
        
    def compSigma(self):
        self.Sigma = self.X.transpose().dot(self.X);
        self.w, self.v = LA.eig(self.Sigma);
        print("PCA : Sigma Computed...")
    
    #def composeInput(self, k):
    #    weights = np.zeros((self.N, k));
    #    for i in range(0, k):
    #        weights[:, i] = np.reshape(self.X.dot(np.reshape(self.v[:,i], (784,1))), self.N);
    #    print("PCA : Data decomposed to " + str(k) + " dimensions...")
    #    return weights
    
    def projPCA(self, procdata, k):
        weights = procdata.dot(self.v[:, 0:k]);
        print("PCA : Data decomposed to " + str(k) + " dimensions...")
        return weights
    
    
    def decompose(self, procdata, k):
        procdata = np.concatenate((procdata, np.zeros((procdata.shape[0], self.d-procdata.shape[1]))), axis = 1); 
        outImage = np.zeros(procdata.shape);
        for i in range(0, k):
            outImage += np.reshape(procdata[:,i], (procdata.shape[0], 1)) * np.reshape(self.v[:,i], (1,784))
        print("PCA : Input data decomposed to " + str(k) + " dimensions...")
        return outImage
    
    """
    def decompose(self, inFeat):
        Feats = np.concatenate((inFeat, np.zeros((inFeat.shape[0], self.d-inFeat.shape[1]))), axis = 1); 
        weights = np.zeros(k);
        outImage = np.zeros(inNumber.shape);
        for i in range(0, k):
            weights[i] = inNumber.transpose().dot(np.reshape(self.v[:,i], (784,1)))
            outImage += weights[i] * np.reshape(self.v[:,i], (784,1))
        return outImage
        return Feats
    """
    
testX = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]));
testY1 = testY

model = PCA();
model.load_train(trainX);
model.compSigma();
pca50Xtr = model.projPCA(model.X, 50);
pca50Xte = model.projPCA(testX, 50);


class neuralnet:
    def __init__(self, lrnRate, connect, nNeurons_FC, batch_size):
        self.connect = connect;
        self.nNeurons_FC = nNeurons_FC;
        self.nLayers = len(connect);
        self.lrnRate = lrnRate;
        self.batchSize = batch_size
        print("NN : Running Neural Net ", [connect[i] for i in range(self.nLayers)])
    
    def load_train(self, trainX, trainY):
        s = trainX.shape;
        self.d = s[1];
        self.N = s[0];
        self.K = np.max(trainY)+1
        #self.trainX = np.insert(trainX, 0, 1, axis=1);
        self.trainX = trainX
        
        self.trainY1 = trainY;
        self.trainY = np.zeros((self.N, self.K),dtype=np.float)
        for i in range(0,self.K):
            temp = np.reshape(np.array(trainY == i, dtype=float), (self.N, 1));
            self.trainY[:,i] = np.asmatrix(np.reshape(temp, self.N))
            
        print("NN : Training data loaded...")
        
    def load_batch(self):
        #print (str(self.batchFrom) + ' ' + str(self.batchTo))
        self.Y1 = self.trainY1[self.perm[self.batchFrom:self.batchTo]]
        self.X = self.trainX[self.perm[self.batchFrom:self.batchTo], :]
        #self.X = np.insert(self.X, 0, 1, axis=1);
        self.Y = self.trainY[self.perm[self.batchFrom:self.batchTo], :]
        self.batchFrom += self.batchSize
        self.batchTo += self.batchSize
        self.batchFrom = (self.batchFrom - self.N) if self.batchFrom >= self.N else self.batchFrom
        self.batchTo = (self.batchTo - self.N) if self.batchTo > self.N else self.batchTo
        if self.batchFrom > self.batchTo or self.batchFrom==0:
            self.newBatch = True;
            self.batchFrom += self.batchSize
            self.batchTo += self.batchSize
            self.batchFrom = (self.batchFrom - self.N) if self.batchFrom >= self.N else self.batchFrom
            self.batchTo = (self.batchTo - self.N) if self.batchTo > self.N else self.batchTo
            self.iterAll = self.iterAll + 1;
            self.perm = np.random.permutation(self.N)
        else:
            self.newBatch = False;
    
    def init_net(self):
        if self.connect[0] is "FC":
            self.layers = [np.random.uniform(-1/np.sqrt(50), 1/np.sqrt(50), (self.nNeurons_FC[0], self.d))];
            self.bias = [np.random.uniform(-1/np.sqrt(50), 1/np.sqrt(50), (1, self.nNeurons_FC[0]))];
        else:
            print("Currently only support when the first layer is FC")
            
        for i in range(1,self.nLayers):
            if self.connect[i] is "FC":
                self.layers.append(np.random.uniform(-1/np.sqrt(50), 1/np.sqrt(50), (self.nNeurons_FC[i], self.nNeurons_FC[i-1])));
                self.bias.append(np.random.uniform(-1/np.sqrt(50), 1/np.sqrt(50), (1, self.nNeurons_FC[i])));
            else:
                self.layers.append(np.zeros(0));
                self.bias.append(np.zeros(0));
        
        self.layervals = []
        for i in range(self.nLayers-1):
            self.layervals.append(np.random.uniform(-1/np.sqrt(50), 1/np.sqrt(50), (self.batchSize, self.nNeurons_FC[i])));
                
        self.gradients = [];
        for i in range(self.nLayers):
            self.gradients.append(np.random.uniform(-1/np.sqrt(50), 1/np.sqrt(50), (self.batchSize, self.nNeurons_FC[i])));
        
        self.iter = 0;
        self.iterAll = 0;
        self.prevLoss = 999999;
        self.weightDist = 99999;
        self.batchFrom = 0;
        self.batchTo = self.batchSize;
        self.perm = range(0, self.N)
        self.loss01Test = [];
        self.lossTest = [];
        self.loss01Tr = [];
        self.lossTr = [];
        print("NN : Neural Net initialized...")
    
    def forward(self):
        curval = self.X;
        for i in range(self.nLayers):
            if self.connect[i] is "FC":
                curval = curval.dot(self.layers[i].transpose()) + np.tile(self.bias[i],(self.batchSize, 1));
                curval[curval > sys.maxint] = sys.maxint
                self.layervals[i] = curval
            elif self.connect[i] is "TANH":
                curval = np.tanh(curval);
                self.layervals[i] = curval
            elif self.connect[i] is "LOSS":
                self.loss = np.sum((curval - self.Y)*(curval - self.Y));
            elif self.connect[i] is "RELU":
                curval = np.maximum(np.zeros(curval.shape), curval);
                self.layervals[i] = curval;
        
        pred = np.argmax(curval, axis = 1);
        correct = np.sum(np.float16(pred == self.Y1))
        self.loss01 = correct / np.float16(self.batchSize);
        #print("NN : forwarded " + str(self.nLayers) + " layers...")
        #print("NN : train loss : " + str(self.loss));
        #print("NN : train 0/1 loss : " + str(self.loss01));
    
    def compGradLoss(self, i):
        self.gradients[i] = 2 * (self.layervals[i-1] - self.Y) / self.batchSize;
        
    def compGradFC(self, i):
        # Forward using function : Y = X * W + b;
        # Backward using function : dL / dX = (dL / dY) * (dY / dX) = (dL / dY) * W.trans(); 
        self.gradients[i] = self.gradients[i+1].dot(self.layers[i]);
        if i==0:
            weightGrad = self.gradients[i+1].transpose().dot(self.X);
        else:
            weightGrad = self.gradients[i+1].transpose().dot(self.layervals[i-1]);
        
        #pdb.set_trace()
        self.layers[i] = self.layers[i] - self.lrnRate * weightGrad;
        
        biasGrad = np.sum(self.gradients[i+1], axis = 0).transpose()
        #biasGrad = self.gradients[i+1].transpose().dot(np.ones((self.batchSize, 1)));
        self.bias[i] = self.bias[i] - self.lrnRate * biasGrad.transpose();
        
    def compGradTanh(self, i):
        self.gradients[i] = self.gradients[i+1] * (1 - np.tanh(self.layervals[i-1])*np.tanh(self.layervals[i-1]));
        
    def compGradReLu(self, i):
        self.gradients[i] = self.gradients[i+1] * np.float16(self.layervals[i-1] > 0);
        
    def forward_all(self, inX, inY):
        curval = inX;
        self.testN = inX.shape[0];
        
        inY1 = inY;
        inY = np.zeros((inY.shape[0], 10),dtype=np.float)
        for i in range(0,10):
            temp = np.reshape(np.array(testY1 == i, dtype=float), (inY.shape[0], 1));
            inY[:,i] = np.asmatrix(np.reshape(temp, inY.shape[0]))
        
        
        layervals = []
        for i in range(self.nLayers):
            if self.connect[i] is "LOSS":
                layervals.append(np.zeros(1));
            else:
                layervals.append(np.zeros((self.testN, self.nNeurons_FC[i])));
        
        
        for i in range(self.nLayers):
            if self.connect[i] is "FC":
                curval = curval.dot(self.layers[i].transpose()) + np.tile(self.bias[i],(self.testN, 1));
                layervals[i] = curval
            elif self.connect[i] is "TANH":
                curval = np.tanh(curval);
                layervals[i] = curval
            elif self.connect[i] is "RELU":
                curval = np.maximum(np.zeros(curval.shape), curval);
                layervals[i] = curval;
            elif self.connect[i] is "LOSS":
                self.lossTest.append(np.sum((curval - inY)*(curval - inY)) / self.testN);
        
        pred = np.argmax(curval, axis = 1);
        correct = np.sum(np.float16(pred == inY1))
        self.loss01Test.append(1 - correct / np.float16(self.testN));
            
        print("NN : test loss : " + str(self.lossTest[-1]));
        print("NN : test 0/1 loss : " + str(self.loss01Test[-1]));
        
    def forward_all_Tr(self, inX, inY):
        curval = inX;
        self.trainN = inX.shape[0];
        
        inY1 = inY;
        inY = np.zeros((inY.shape[0], 10),dtype=np.float)
        for i in range(0,10):
            temp = np.reshape(np.array(self.trainY1 == i, dtype=float), (inY.shape[0], 1));
            inY[:,i] = np.asmatrix(np.reshape(temp, inY.shape[0]))
        
        
        layervals = []
        for i in range(self.nLayers):
            if self.connect[i] is "LOSS":
                layervals.append(np.zeros(1));
            else:
                layervals.append(np.zeros((self.trainN, self.nNeurons_FC[i])));
        
        
        for i in range(self.nLayers):
            if self.connect[i] is "FC":
                curval = curval.dot(self.layers[i].transpose()) + np.tile(self.bias[i],(self.trainN, 1));
                layervals[i] = curval
            elif self.connect[i] is "TANH":
                curval = np.tanh(curval);
                layervals[i] = curval
            elif self.connect[i] is "RELU":
                curval = np.maximum(np.zeros(curval.shape), curval);
                layervals[i] = curval;
            elif self.connect[i] is "LOSS":
                self.lossTr.append(np.sum((curval - inY)*(curval - inY)) / self.trainN);
        
        pred = np.argmax(curval, axis = 1);
        correct = np.sum(np.float16(pred == inY1))
        self.loss01Tr.append(1 - correct / np.float16(self.trainN));
            
        print("NN : train loss : " + str(self.lossTr[-1]));
        print("NN : train 0/1 loss : " + str(self.loss01Tr[-1]));
    
    def getRandNode(self, k):
        self.perm = np.random.permutation(self.nNeurons_FC[0])[0:10]
        feats = self.layers[0][self.perm,:]
        
        return feats;
        
#connect = ["FC", "TANH", "FC", "LOSS"];
connect = ["FC", "RELU", "FC", "RELU", "LOSS"];
nNeurons_FC = [500, 500, 10, 10, 1];

NN = neuralnet(0.01, connect, nNeurons_FC, 5);

NN.load_train(pca50Xtr, trainY);
NN.init_net();

NN.iter = 12000 * 30
#NN.iter = 12000
NN.epoch = 1;
while NN.iter>0:
    #if NN.iter>11600:
    #    pdb.set_trace()
    
    NN.load_batch();
    NN.forward();
    NN.compGradLoss(3)
    NN.compGradFC(2)
    #pdb.set_trace()
    NN.compGradTanh(1)
    NN.compGradFC(0)
    #pdb.set_trace()
    NN.forward();
    if NN.iter%6000 == 0:
        print("Processing epoch " + str(np.float16(NN.epoch)/2))
        NN.forward_all(pca50Xte, testY);
        NN.forward_all_Tr(pca50Xtr, trainY);
        print("\n")
        NN.epoch += 1;
        #pdb.set_trace()
    if NN.iter == 12000 * 15:
        NN.lrnRate /= 5;
    
    NN.iter -= 1;
    
    
matplotlib.pyplot.clf()
length = len(NN.loss01Tr)
matplotlib.pyplot.ylim([0,0.07])
matplotlib.pyplot.title("Plot of 0/1 loss with Tanh transfer function")
red_star = matplotlib.pyplot.plot(range(1, length+1), NN.loss01Tr[0:length], color="blue", hold = True, linewidth=2.0)
red_star = matplotlib.pyplot.plot(range(1, length+1), NN.loss01Test[0:length], color="red", hold = True, linewidth=2.0)
matplotlib.pyplot.savefig('relu2loss01.png', transparent = True)


matplotlib.pyplot.clf()
length = len(NN.lossTr)
matplotlib.pyplot.ylim([0,0.5])
matplotlib.pyplot.title("Plot of square loss with Tanh transfer function")
red_star = matplotlib.pyplot.plot(range(1, length+1), NN.lossTr[0:length], color="blue", hold = True, linewidth=2.0)
red_star = matplotlib.pyplot.plot(range(1, length+1), NN.lossTest[0:length], color="red", hold = True, linewidth=2.0)
matplotlib.pyplot.savefig('relu2loss.png', transparent = True)

def plotMNIST(feats, shape, name):
    Image = np.zeros((28*shape[0], 28*shape[1]), dtype=np.uint8)
    for i in range(feats.shape[0]):
        x = i / 5
        y = i % 5
        temp = np.reshape(feats[i,:], (28, 28)) * 255.0 *20
        temp[temp<0] = 0
        temp[temp>255] = 255
        Image[x*28:(x+1)*28, y*28:(y+1)*28] = np.uint8(temp)
    cv2.imwrite(name, Image);
feats = NN.getRandNode(10);
feats = model.decompose(feats, 50);
plotMNIST(feats, (2, 5), 'relu2_node10.png')