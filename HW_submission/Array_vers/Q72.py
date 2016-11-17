from Lasso import lasso
from tempfile import TemporaryFile
import numpy as np

model = lasso(1.5, 0.1, 0.985);

model.load_data("training");
model.fit(True);

outfile = TemporaryFile()
np.savez(outfile, model.w)

print("Training Accuracy:")
model.predict();

model.load_data("testing");
print("Testing Accuracy:")
model.predict();