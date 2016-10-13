from Lasso import lasso
from tempfile import TemporaryFile
import numpy as np

model = lasso(2.0, 0.00001, 0.985);
model.generate_data(1000, 20, 5, 0.0)

model.fit();

outfile = TemporaryFile()
np.savez(outfile, model.w)

print("Training Accuracy:")
model.predict();

print("Testing Accuracy:")
model.generate_data(1000, 20, 5, 0.2)
model.predict();

for x in range(0,20):
    print(model.w[x])
