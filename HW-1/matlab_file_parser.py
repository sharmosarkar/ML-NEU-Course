
import scipy.io
import numpy as np

data = scipy.io.loadmat("Z:\ML\HW-1\linear_regression.mat")
print(data)

for i in data:
    if '__' not in i and 'readme' not in i:
          np.savetxt(("Z:\ML\HW-1\\file.csv"),data[i],delimiter=',')