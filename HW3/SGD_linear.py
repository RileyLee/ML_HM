import numpy as np
import matplotlib.pyplot
import mnist
import scipy
import scipy.sparse.linalg
from scipy.sparse import csr_matrix
from scipy.sparse import identity
import pdb
from scipy.linalg import norm
from scipy.spatial.distance import pdist

trainX, trainY = mnist.load_mnist("training", None, './MNIST');
testX, testY = mnist.load_mnist("testing", None, './MNIST');

print(str(trainX.shape))
print("MNIST loaded")

class SGDRBF:
    
    
    def __init__(self, eta, lamda, thresh, batchSize, kernelBW):
        self.eta = eta;
        self.lamda = lamda;
        self.thresh = thresh;
        self.regularized = True;
        self.lossSet = np.zeros(100000);
        self.testLossSet = np.zeros(100000);
        self.testLossSetAve = np.zeros(100000);
        self.loss01SetTr = np.zeros(100000);
        self.loss01SetTe = np.zeros(100000);
        self.loss01SetTeAve = np.zeros(100000);
        self.trainLossSet = np.zeros(100000);
        self.trainLossSetAve = np.zeros(100000);
        self.loss01SetTrAve = np.zeros(100000);
        self.batchSize = batchSize;
        self.kernelBW = kernelBW;
        self.stoch = True;
        
    def load_train(self, trainX, trainY):
        self.trainY1 = trainY;
        s = trainX.shape;
        self.d = s[1] * s[2] + 1;
        #self.d = s[1] + 1;
        self.n = self.d-1;
        self.N = s[0];
        self.trainX = np.asmatrix(np.reshape(trainX, (self.N, self.d-1)));
        #pdb.set_trace()
        self.K = np.max(trainY) + 1
        self.trainY = np.zeros((self.N, self.K),dtype=np.float)
  
        for i in range(0,self.K):
            temp = np.reshape(np.array(trainY == i, dtype=float), (self.N, 1));
            self.trainY[:,i] = np.asmatrix(np.reshape(temp, self.N))
            
        if self.stoch==False:
            self.probTr = np.asmatrix(np.zeros((self.N, self.K),dtype=float))
        else:
            self.probTr = np.asmatrix(np.zeros((self.batchSize, self.K),dtype=float))
        self.probTr1 = np.asmatrix(np.zeros((self.N, self.K),dtype=float))
        print("Training data loaded...")
        
    def load_test(self, testX, testY):
        self.testY1 = testY;
        #pdb.set_trace()
        self.testN = testX.shape[0];
        self.testX = np.asmatrix(np.reshape(testX, (self.testN, self.d-1)));
        #pdb.set_trace()
        self.testX = np.sin(self.testX * self.rand_kernel / self.kernelBW)
        self.testX = np.insert(self.testX, 0, 1, axis=1);
        
        self.testY = np.zeros((self.testN, self.K),dtype=np.float)
        self.probTe = np.asmatrix(np.zeros((self.testN, self.K),dtype=float))
        
        for i in range(0,self.K):
            temp = np.reshape(np.array(testY == i, dtype=float), (self.testN, 1));
            self.testY[:,i] = np.asmatrix(np.reshape(temp, self.testN))
        print("Testing data loaded...")
            
    def load_batch(self):
        #print (str(self.batchFrom) + ' ' + str(self.batchTo))
        self.Y1 = self.trainY1[self.perm[self.batchFrom:self.batchTo]]
        self.X = np.sin(self.trainX[self.perm[self.batchFrom:self.batchTo], :] * self.rand_kernel / self.kernelBW)
        self.X = np.insert(self.X, 0, 1, axis=1);
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
        
        
    def initIteration(self):
        self.weights = np.asmatrix(np.zeros((self.N + 1, self.K), dtype=float));
        self.iter = 0;
        self.iterAll = 0;
        self.prevLoss = 999999;
        self.weightDist = 99999;
        self.batchFrom = 0;
        self.batchTo = self.batchSize;
        self.perm = range(0, self.N)
        
    
    def computeTrainLoss(self):
        trainLoss = 0;  trainLossAve = 0;
        total = 0; batch = 5000; totalAve = 0;
        for i in range(0,self.N/batch):
            print("Evaluating training accuracy : Part " + str(i+1) + "....")
            Y1 = self.trainY1[i*batch : i*batch+1000]
            X = np.sin(self.trainX[i*batch : i*batch+1000, :] * self.rand_kernel / self.kernelBW)
            X = np.insert(X, 0, 1, axis=1);
            Y = self.trainY[i*batch : i*batch+1000, :]
            pred_y = X * self.linearWeight
            pred_yAve = X * self.weightAve
            predLabelTr = np.argmax(pred_y, axis=1);
            predLabelTrAve = np.argmax(pred_yAve, axis=1);
            trainLoss += np.sum(np.square(Y - pred_y));
            trainLossAve += np.sum(np.square(Y - pred_yAve));
            
            temp = (np.array(predLabelTr) != np.reshape(Y1, (batch/5, 1))).astype(float)
            total += np.sum(temp);
            temp = (np.array(predLabelTrAve) != np.reshape(Y1, (batch/5, 1))).astype(float)
            totalAve += np.sum(temp);
            #pdb.set_trace()
        self.loss01SetTr[self.iterA] = np.float(total) / self.N * 5; 
        self.loss01SetTrAve[self.iterA] = np.float(totalAve) / self.N * 5; 
        
        self.trainLossSet[self.iterA] = trainLoss /  2 / self.N * 5 + self.lamda * norm(self.weightAve, 2);
        self.trainLossSetAve[self.iterA] = trainLossAve /  2 / self.N * 5 + self.lamda * norm(self.weightAve, 2);
    
    def computeTestLoss(self):
        self.predT_y = self.testX * self.linearWeight;
        self.predT_yAve = self.testX * self.weightAve;
        self.testLossSet[self.iterA] = np.sum(np.square(self.testY - self.predT_y)) / 2 / self.testN + self.lamda * norm(self.weightAve, 2);
        self.testLossSetAve[self.iterA] = np.sum(np.square(self.testY - self.predT_yAve)) / 2 / self.testN + self.lamda * norm(self.weightAve, 2);
        self.predLabelTe = np.argmax(self.predT_y, axis=1);
        self.predLabelTeAve = np.argmax(self.predT_yAve, axis=1);
        temp = (np.array(self.predLabelTe) != np.reshape(self.testY1, (self.testN, 1))).astype(float)
        self.loss01SetTe[self.iterA] = np.sum(temp) / self.testN;
        temp = (np.array(self.predLabelTeAve) != np.reshape(self.testY1, (self.testN, 1))).astype(float)
        self.loss01SetTeAve[self.iterA] = np.sum(temp) / self.testN;
        


    def linearfit(self, flag_print_status, flag_eval, trainY):

        # Initialization
        self.linearWeight = np.asmatrix(np.zeros( (self.N+1, self.K) ));
        self.iter = 0;
        self.iterA = 0;
        converge = False;
        
        self.load_batch();
        
        self.pred_y = self.X * self.linearWeight;
        self.weightAve = np.zeros((self.N+1, 10));
        
        # Start Iterations
        idx = 0;
        while not converge:
            if (self.iter>6000):
                self.eta = 0.0001;
            
            #if (flag_print_status):
            #            print ("Processing Epoch " + str(self.iter));
            
            if self.iter > 0:
                self.load_batch();
            self.iter += 1;
            
            #pdb.set_trace()
            self.pred_y = self.X * self.linearWeight;
            gradient = - self.X.transpose() * (np.asmatrix(self.Y) - self.pred_y) / self.batchSize;
            gradient = gradient + 2 * self.lamda * self.linearWeight;
            self.linearWeight = self.linearWeight - self.eta * gradient;
            self.weightAve += self.linearWeight;
            
            self.trainLoss = np.sum(np.square(self.Y - self.pred_y)) / 2 / self.batchSize + self.lamda * norm(self.linearWeight, 2);
            #print("\tTraining loss : " + str(self.trainLoss))
            idx += 1;
            
            if self.newBatch or np.floor(np.float(self.batchFrom)/20000)==np.float(self.batchFrom)/20000:
                self.weightAve = self.weightAve / idx;
                model.computeTrainLoss()
                model.computeTestLoss()
                self.iterA += 1;
                idx = 0;
                if (flag_print_status):
                    pdb.set_trace()
                    print ("Processing Iteration " + str(self.iter));
                    print("Training loss : " + str(self.trainLossSet[self.iterA]))
                    print("Testing loss : " + str(self.testLossSet[self.iterA]))
                    print("Training 0/1 loss : " + str(self.loss01SetTr[self.iterA]))
                    print("Testing 0/1 loss : " + str(self.loss01SetTe[self.iterA]))
                    print("Training 0/1 Ave loss : " + str(self.loss01SetTrAve[self.iterA]))
                    print("Testing 0/1 Ave loss : " + str(self.loss01SetTeAve[self.iterA]))
                    print("Training Ave loss : " + str(self.trainLossSetAve[self.iterA]))
                    print("Testing Ave loss : " + str(self.testLossSetAve[self.iterA]))

            converge = False if self.trainLoss > self.thresh or self.iter < 3 or self.iterAll<4 else True;
            
    
    def generateWeight(self, sigma):
        self.rand_kernel = np.asmatrix(np.random.normal(0, sigma, (self.n, self.N)));
        print("Kernel generated")
        
model = SGDRBF(0.0005, 0, 0.02, 10, 3.76);

model.load_train(trainX, trainY);
model.generateWeight(1.0);
model.load_test(testX, testY);

model.initIteration();

model.linearfit(True, 2, trainY);

matplotlib.pyplot.clf()
length = sum(model.loss01SetTeAve>0)
red_star = matplotlib.pyplot.plot(range(1, length+1), model.trainLossSet[0:length], color="blue", hold = True, linewidth=2.0)
red_star = matplotlib.pyplot.plot(range(1, length+1), model.testLossSet[0:length], color="red", hold = True, linewidth=2.0)
red_star = matplotlib.pyplot.plot(range(1, length+1), model.trainLossSetAve[0:length], color="green", hold = True, linewidth=2.0)
red_star = matplotlib.pyplot.plot(range(1, length+1), model.testLossSetAve[0:length], color="purple", hold = True, linewidth=2.0)
matplotlib.pyplot.savefig('Q212.png', transparent = True)

matplotlib.pyplot.clf()
length = sum(model.loss01SetTeAve>0)
red_star = matplotlib.pyplot.plot(range(1, length+1), model.loss01SetTr[0:length], color="blue", hold = True, linewidth=2.0)
red_star = matplotlib.pyplot.plot(range(1, length+1), model.loss01SetTe[0:length], color="red", hold = True, linewidth=2.0)
red_star = matplotlib.pyplot.plot(range(1, length+1), model.loss01SetTrAve[0:length], color="green", hold = True, linewidth=2.0)
red_star = matplotlib.pyplot.plot(range(1, length+1), model.loss01SetTeAve[0:length], color="purple", hold = True, linewidth=2.0)
matplotlib.pyplot.savefig('Q213.png', transparent = True)