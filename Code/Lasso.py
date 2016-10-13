from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import mnist
import numpy as np

class lasso:

    def __init__(self, lamda, thresh, decay):
        self.lamda = lamda;
        self.thresh = thresh
        self.decay = decay


    def load_data(self, string):
        if string is "training":
            ori_X, Y = mnist.load_mnist("training", None, '../hw1-data');
        elif string is "testing":
            ori_X, Y = mnist.load_mnist("testing", None, '../hw1-data');

        s = ori_X.shape;
        self.n = s[1] * s[2];
        self.N = s[0];
        #self.X = np.reshape(ori_X, (len(Y), self.n));
        #self.Y = np.array(Y == 2, dtype=float);
        #self.Y = np.reshape(self.Y, (self.N,1))

        self.X_sparse = csc_matrix(np.reshape(ori_X, (self.N, self.n)), shape=(self.N, self.n))
        self.Y_sparse = csc_matrix(np.reshape(np.array(Y == 2, dtype=float), (self.N,1)), shape=(self.N, 1))

        print("Loading data " + string + " complete....")

    def mountData(self, X, Y):
        self.X = X;
        self.Y = Y;
        print("Here")
        s = self.X.shape;
        self.N = s[0];
        self.n = s[1];
        self.Y = np.reshape(self.Y, (self.N, 1))
        self.X_sparse = csc_matrix(self.X, shape=(self.N, self.n))#.toarray()
        self.Y_sparse = csc_matrix(self.Y, shape=(self.N, 1))#.toarray()

    def mountName(self, featName):
        self.featName = featName;

    def fit(self):

        # Initialization
        #self.w = np.zeros( (self.n, 1) );
        self.w = csc_matrix(np.zeros( (self.n, 1) ), shape=(self.n, 1));
        self.w0 = 0;
        self.iter = 0;
        converge = False;

        # Start Iterations
        while not converge:
            self.iter += 1
            print ("Processing Iteration " + str(self.iter));
            self.w_prev = csc_matrix.copy(self.w)
            self.w0_prev = self.w0;
            self.pred_y = self.X_sparse.dot(self.w) + self.w0;
            self.w0 = (self.Y_sparse - self.pred_y).sum(axis=0) / self.N + self.w0
            self.pred_y = self.pred_y - self.w0_prev + self.w0;

            for k in range(0, self.n, 1):
                print (str(k))
                c = self.X_sparse[:, k]
                #c = 2 * self.X_sparse[:, k].transpose().dot(self.Y_sparse - self.pred_y + self.X_sparse[:, k].dot(self.w[k]))
                #a = 2 * self.X_sparse[:, k].transpose().dot(self.X_sparse[:, k])

                #c = 2 * np.dot(np.transpose(self.X_sparse[:,k]), (self.Y_sparse - self.pred_y + np.reshape(self.w[k] * self.X_sparse[:,k], (self.N, 1))))
                #a = 2 * np.dot(np.transpose(self.X_sparse[:,k]), self.X_sparse[:,k]);
                """

                if c < -self.lamda:
                    self.w[k, 0] = (c[0, 0] + self.lamda) / a[0, 0]
                elif c > self.lamda:
                    self.w[k, 0] = (c[0, 0] - self.lamda) / a[0, 0]
                else:
                    self.w[k, 0] = 0

                """
                #self.pred_y = self.pred_y + self.X_sparse[:, k].dot(self.w[k] - self.w_prev[k])
                #self.pred_y = self.pred_y + np.reshape(self.X_sparse[:, k] * (self.w[k] - self.w_prev[k]), (self.N, 1))
                #self.pred_y = self.pred_y + np.reshape(self.X_sparse[:,k] * (self.w[k] - self.w_prev[k]), (self.N, 1))


            abs_sum = max(abs(self.w_prev - self.w));

            pred = np.dot(self.X, self.w) + self.w0;
            #print("Error : " + str(sum(abs(self.pred_y - pred))))
            RMSE = np.sqrt(np.sum(np.power(self.Y - pred, 2)) / self.N);

            print("Absolute sum : " + str(abs_sum))
            print("RMSE : " + str(RMSE) + "   Leanring rate : " + str(self.lamda))

            self.lamda *= self.decay;

            converge = False if abs_sum > self.thresh else True;


    def predict_hard(self):
        pred = np.dot(self.X, self.w) + self.w0;
        hard_pred = np.array(pred > 0.5, dtype=float)
        Err = np.sum(abs(self.Y - hard_pred)) / self.N;
        Err_squre = np.sum(np.power(self.Y - pred, 2)) / self.N;
        print ("0/1 loss : " + str(Err));
        print ("square loss : " + str(Err_squre));
        return Err, Err_squre

    def predict(self):
        pred = np.dot(self.X, self.w) + self.w0;
        Err = np.sum(abs(self.Y - pred)) / self.N;
        Err_squre = np.sum(np.power(self.Y - pred, 2)) / self.N;
        RMSE = np.sqrt(np.sum(np.power(self.Y - pred, 2)) / self.N);
        print ("0/1 loss : " + str(Err));
        print ("square loss : " + str(Err_squre));
        print ("RMSE : " + str(RMSE))
        return Err, Err_squre, RMSE

    def generate_data(self, N, d, k, sigma):
        self.N = N;
        self.n = d;

        weight = np.ones((k,1)) * 10
        weight = np.concatenate((weight, np.zeros((d-k,1))),axis=0)
        weight0 = 0
        rand = np.random.randn(N, d);
        self.X = np.copy(rand)
        self.Y = weight0 + np.dot(rand, weight)
        rand = np.random.randn(N, 1)
        self.Y = self.Y + rand * sigma;
        self.Y = np.reshape(self.Y, (self.N, 1))
        self.X_sparse = csr_matrix(self.X, shape=(self.N, self.n )).toarray()
        self.Y_sparse = csr_matrix(self.Y, shape=(self.N, 1)).toarray()


