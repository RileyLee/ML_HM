from Lasso import lasso
from tempfile import TemporaryFile
import numpy as np
import matplotlib.pyplot

num_pts = 80;

recall = np.zeros(num_pts);
prec = np.zeros(num_pts);
X = np.zeros(num_pts);

idx = 0;
lamda = 0.5
for i in range(1, num_pts+1, 1):
    lamda = lamda * 1.1
    print ("Running Lambda " + str(lamda))
    model = lasso(lamda, 0.00001, 0.985)
    model.generate_data(50, 75, 5, 1.0)
    model.fit(False, 1);

    recall[idx], prec[idx] = model.evalRP();
    X[idx] = lamda
    idx += 1;

matplotlib.pyplot.plot(X, recall, color="red", hold = True)
matplotlib.pyplot.plot(X, prec, hold = True, color="blue")
matplotlib.pyplot.ylim([0,1.1])
matplotlib.pyplot.savefig("out.svg", transparent = True)
