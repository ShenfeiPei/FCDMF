import numpy as np
import time
from scipy import sparse
import sys
from FCDMF import FCDMF
from Public import Ifuns, Gfuns, Mfuns

X, y_true, N, dim, c_true = Ifuns.load_mat("./dataset/ORL32x32.mat")
#  X = Ifuns.normalize_fea(X, 0)
print(N, dim, c_true)

num_anchor = int(min(N / 2, 1024))
num_anchor = int(max(num_anchor, c_true + 5))
anchor_way = "k-means++"
Anchor = Gfuns.get_anchor(X=X, m=num_anchor, way=anchor_way)

graph_knn = np.minimum(2 * c_true, num_anchor - 1)
graph_way = "t_free"
B = Gfuns.kng_anchor(X=X, knn=graph_knn, way=graph_way, Anchor=Anchor)

t1 = time.time()
obj = FCDMF(B.astype(np.float64), c_true)
obj.opt(rep=10, ITER=100, init="k-means++")
acc = Mfuns.multi_accuracy(y_true, obj.y_pre)
t2 = time.time()
print(np.mean(acc), "time = ", t2 - t1)


# paper, ORL32, acc = 0.538
# run,   ORL32, acc = 0.517
