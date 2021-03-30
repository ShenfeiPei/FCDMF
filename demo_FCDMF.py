import numpy as np
import time
from scipy import sparse
import sys
from FCDMF2.FCDMF import FCDMF
from FCDMF2.Public import Ifuns, Gfuns, Mfuns

X, y_true, N, dim, c_true = Ifuns.load_mat("./dataset/BinaryAlpha_20200916.mat")
#  X = Ifuns.normalize_fea(X, 0)
print(N, dim, c_true)

num_anchor = int(min(N / 2, 1024))
anchor_way = "k-means++"
Anchor = Gfuns.get_anchor(X=X, m=num_anchor, way=anchor_way)

graph_knn = np.minimum(2 * c_true, num_anchor)
graph_way = "t_free"
B = Gfuns.kng_anchor(X=X, knn=graph_knn + 1, way=graph_way, Anchor=Anchor)

t1 = time.time()
obj = FCDMF(B.astype(np.float64), c_true)
obj.opt(rep=10, ITER=100, init="k-means++")
acc = Mfuns.multi_accuracy(y_true, obj.y_pre)
t2 = time.time()
print(np.mean(acc), "time = ", t2 - t1)


#  paper, Binalpha, acc = 0.421
#  run,   Binalpha, acc = 0.441
