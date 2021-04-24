import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image


def process(f):
    im = np.array(Image.open(f)) / 255
    im_reshape = im.reshape((-1, 3))
    return np.mean(im_reshape, axis=0), np.std(im_reshape, axis=0)


datadir = sys.argv[1]
files = list(Path(datadir).glob("./**/*.png"))

out = Parallel(n_jobs=-1, verbose=10)(delayed(process)(f) for f in files)
mean = np.mean(np.stack([e[0] for e in out]), axis=0)
sigma = np.mean(np.stack([e[1] for e in out]), axis=0)

np.savetxt("mean.txt", mean)
np.savetxt("sigma.txt", sigma)
