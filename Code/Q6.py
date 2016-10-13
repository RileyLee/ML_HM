import mnist
import numpy as np

def eval(X, Y, w, param):
    mX = np.asmatrix(X);
    mY = np.asmatrix(Y);
    mw = np.asmatrix(w);
    cost = mX * mw.transpose() - mY.transpose()
    print(cost.shape)
    cost = np.sum(np.power(cost, 2)) / len(Y) + param['lamda'] * np.sum(np.power(w, 2));
    print (cost)
    return cost

def classify(mX, w):
    return mX * w;

class linear:

    lamda = 1.0;
    X = [];
    Y = [];
    w = [];
    N = n = 0;

    def __init__(self, lamda):
        self.lamda = lamda

    def load_train_data(self):
        ori_X, Y = mnist.load_mnist("training", None, '../hw1-data');
        s = ori_X.shape;
        self.n = s[1] * s[2];
        self.N = s[0];
        X = np.reshape(ori_X, (len(Y), self.n));
        self.X = np.insert(X, 0, 1, axis=1);
        self.Y = np.array(Y == 2, dtype=float);
        print("Data loading complete....")

    def load_test_data(self):
        ori_X, Y = mnist.load_mnist("testing", None, '../hw1-data');
        s = ori_X.shape;
        self.n = s[1] * s[2];
        self.N = s[0];
        X = np.reshape(ori_X, (len(Y), self.n));
        self.X = np.insert(X, 0, 1, axis=1);
        self.Y = np.array(Y == 2, dtype=float);
        print("Data loading complete....")

    def fit(self):
        sigma = np.dot(np.transpose(self.X), self.X) / self.N;
        left = np.linalg.inv(sigma + self.lamda * np.identity(self.n+1))
        right = np.dot(np.transpose(self.X), np.transpose(self.Y)) / self.N;
        self.w = np.dot(left, right);
        print("Training finished.....")

    def predict(self):
        pred = np.dot(self.X, self.w);
        hard_pred = np.array(pred > 0.5, dtype=float)
        Err = np.sum(abs(self.Y - hard_pred)) / self.N;
        Err_squre = np.sum(np.power(self.Y - pred, 2)) / self.N;
        print ("0/1 loss : " + str(Err));
        print ("square loss : " + str(Err_squre));
        return Err, Err_squre


model = linear(1.0);

model.load_train_data();
model.fit();

print("Training Accuracy:")
model.predict();
model.load_test_data();

print("Testing Accuracy:")
model.predict();

