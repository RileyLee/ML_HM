from scipy.sparse import csr_matrix
import matplotlib.pyplot
import mnist
import numpy as np

class lasso:

    def __init__(self, lamda, thresh, decay):
        self.lamda = lamda;
        self.thresh = thresh
        self.decay = decay

        self.recall = np.zeros(2000);
        self.precision = np.zeros(2000);
        self.lamdavals = np.zeros(2000);
        self.nonzeros = np.zeros(2000);
        self.RMSE = np.zeros(2000);
        self.RMSE_val = np.zeros(2000);




    def load_data(self, string):
        if string is "training":
            ori_X, Y = mnist.load_mnist("training", None, '../hw1-data');
        elif string is "testing":
            ori_X, Y = mnist.load_mnist("testing", None, '../hw1-data');

        s = ori_X.shape;
        self.n = s[1] * s[2];
        self.N = s[0];
        self.X = np.reshape(ori_X, (len(Y), self.n));
        self.Y = np.array(Y == 2, dtype=float);
        self.Y = np.reshape(self.Y, (self.N,1))

        self.X_sparse = csr_matrix(self.X, shape=(self.N, self.n)).toarray()
        self.Y_sparse = csr_matrix(self.Y, shape=(self.N, 1)).toarray()

        print("Loading data " + string + " complete....")

    def mountData(self, X, Y, string):
        if string is "training":
            self.X = X;
            self.Y = Y;
            print("Here")
            s = self.X.shape;
            self.N = s[0];
            self.n = s[1];
            self.Y = np.reshape(self.Y, (self.N, 1))
            self.X_sparse = csr_matrix(self.X, shape=(self.N, self.n)).toarray()
            self.Y_sparse = csr_matrix(self.Y, shape=(self.N, 1)).toarray()
        elif string is "validation":
            self.X_val = X;
            self.Y_val = Y;
            print("Here")
            s = self.X_val.shape;
            self.N_val = s[0];
            self.Y_val = np.reshape(self.Y_val, (self.N_val, 1))
        elif string is "testing":
            self.X_te = X;
            self.Y_te = Y;
            print("Here")
            s = self.X_te.shape;
            self.N_te = s[0];
            self.Y_te = np.reshape(self.Y_te, (self.N_te, 1))


    def mountName(self, featName):
        self.featName = featName;

    def fit(self, flag_print_status, flag_eval):

        # Initialization
        self.w = np.zeros( (self.n, 1) );
        self.w0 = 0;
        self.iter = 0;
        self.w_history = np.zeros((2000, self.n));
        converge = False;
        if flag_eval==1:
            self.evalRP();

        # Start Iterations
        while not converge:
            self.iter += 1
            if (flag_print_status):
                print ("Processing Iteration " + str(self.iter));
            self.w_prev = np.copy(self.w);
            self.w0_prev = self.w0;
            self.pred_y = np.dot(self.X_sparse, self.w) + self.w0;
            self.w0 = np.sum(self.Y_sparse - self.pred_y) / self.N + self.w0
            self.pred_y = self.pred_y - self.w0_prev + self.w0;

            for k in range(0, self.n, 1):
                c = 2 * np.dot(np.transpose(self.X_sparse[:,k]), (self.Y_sparse - self.pred_y + np.reshape(self.w[k] * self.X_sparse[:,k], (self.N, 1))))
                a = 2 * np.dot(np.transpose(self.X_sparse[:,k]), self.X_sparse[:,k]);
                if c < -self.lamda:
                    self.w[k] = (c + self.lamda) / a;
                elif c > self.lamda:
                    self.w[k] = (c - self.lamda) / a;
                else:
                    self.w[k] = 0
                self.pred_y = self.pred_y + np.reshape(self.X_sparse[:,k] * (self.w[k] - self.w_prev[k]), (self.N, 1))

            if flag_eval==1:
                self.evalRP();
            elif flag_eval==2:
                self.evalRMSE();
                self.evalnonzeros();
                self.lamdavals[self.iter] = self.lamda
                self.w_history[self.iter, :] = self.w.T;


            abs_sum = max(abs(self.w_prev - self.w));

            pred = np.dot(self.X, self.w) + self.w0;
            RMSE = np.sqrt(np.sum(np.power(self.Y - pred, 2)) / self.N);




            if (flag_print_status):
                print("Absolute sum : " + str(abs_sum))
                print("RMSE : " + str(RMSE) + "   Leanring rate : " + str(self.lamda))


            self.lamda *= self.decay;

            converge = False if abs_sum > self.thresh and self.lamda > 0.001 else True;


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

    def evalRP(self):
        TP = (self.w > 0) * (self.weight > 0)
        TP.astype(float)
        TP = np.sum(TP)
        Pos = self.w > 0
        Pos.astype(float)
        Pos = np.sum(Pos)
        True = self.weight > 0
        True.astype(float)
        True = np.sum(True)
        if (Pos!=0):
            self.precision[self.iter] = np.float(TP) / np.float(Pos)
        else:
            self.precision[self.iter] = 0
        self.recall[self.iter] = np.float(TP) / np.float(True)
        self.lamdavals[self.iter] = self.lamda

    def evalRMSE(self):
        pred = np.dot(self.X, self.w) + self.w0;
        self.RMSE[self.iter] = np.sqrt(np.sum(np.power(self.Y - pred, 2)) / self.N);

        pred = np.dot(self.X_val, self.w) + self.w0;
        self.RMSE_val[self.iter] = np.sqrt(np.sum(np.power(self.Y_val - pred, 2)) / self.N_val);


    def evalRMSE_te(self, lamda):
        idx = (np.abs(self.lamdavals - lamda)).argmin()
        self.lamda = self.lamdavals[idx]
        w = np.reshape(self.w_history[idx, :], (self.n,1))
        pred = np.dot(self.X_te, w) + self.w0;
        self.RMSE_te = np.sqrt(np.sum(np.power(self.Y_te - pred, 2)) / self.N);


    def evalnonzeros(self):
        self.nonzeros[self.iter] = len(np.nonzero(self.w)[0])


    def generate_data(self, N, d, k, sigma):
        self.N = N;
        self.n = d;

        self.weight = np.ones((k,1)) * 10
        self.weight = np.concatenate((self.weight, np.zeros((d-k,1))),axis=0)
        self.weight0 = 0
        rand = np.random.randn(N, d);
        self.X = np.copy(rand)
        self.Y = self.weight0 + np.dot(rand, self.weight)
        rand = np.random.randn(N, 1)
        self.Y = self.Y + rand * sigma;
        self.Y = np.reshape(self.Y, (self.N, 1))
        self.X_sparse = csr_matrix(self.X, shape=(self.N, self.n )).toarray()
        self.Y_sparse = csr_matrix(self.Y, shape=(self.N, 1)).toarray()



