from Lasso import lasso
from tempfile import TemporaryFile
import numpy as np
import matplotlib.pyplot

num_pts = 100;

recall = np.zeros(num_pts);
prec = np.zeros(num_pts);
X = np.zeros(num_pts);

idx = 0;
for i in range(1, num_pts+1, 1):
    lamda = i * 0.1
    print ("Running Lambda " + str(lamda))
    model = lasso(500, 0.001, 0.985)
    model.generate_data(50, 20, 5, 1.0)
    model.fit(False);

    outfile = TemporaryFile()
    np.savez(outfile, model.w)
    recall[idx], prec[idx] = model.evalRP();
    X[idx] = lamda
    idx += 1;

matplotlib.pyplot.plot(X, recall, color="red", hold = True)
matplotlib.pyplot.plot(X, prec, hold = True, color="blue")
matplotlib.pyplot.ylim([0,1.1])
matplotlib.pyplot.savefig("out.svg", transparent = True)
