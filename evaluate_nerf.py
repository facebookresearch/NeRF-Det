import numpy as np
import os

path = 'psnr'
files = os.listdir(path)
l = len(files)
psnr = np.zeros(3)
d = 0
for f in files:
    psnr_ = np.load(os.path.join(path, f), allow_pickle=True)
    d += psnr_[-1].mean()
print(d/l)
