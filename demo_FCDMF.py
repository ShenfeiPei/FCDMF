import numpy as np
import time
from scipy import sparse
import sys
from FCDMF_eigen import FCDMF
from Public import Funs, Gfuns, Mfuns
from scipy.sparse.linalg import svds


X, y_true, N, dim, c_true = Funs.loadmat("./dataset/BinaryAlpha_20200916.mat")

t_start = time.time()

num_anchor = int(min(N / 2, 1024))
anchor_way = "k-means++"
Anchor = Gfuns.get_anchor(X=X, m=num_anchor, way=anchor_way)

graph_knn = np.minimum(2 * c_true, num_anchor)
graph_way = "t_free"
B = Gfuns.kng_anchor(X=X, knn=graph_knn + 1, way=graph_way, Anchor=Anchor)

obj = FCDMF(B.astype(np.float64), c_true)

rep = 10
B_sp = sparse.csr_matrix(B)
U, S, VH = svds(B_sp, k=c_true, which="LM")
P = Funs.initialY(X=U[:, :c_true], c_true=c_true, rep=rep, way="k-means++")
V = VH.T
Q = Funs.initialY(X=V[:, :c_true], c_true=c_true, rep=rep, way="k-means++")

t_end = time.time()
t0 = t_end - t_start

obj.opt(init_P=P, init_Q=Q, ITER=100)
t1 = obj.time_arr

acc = Mfuns.multi_accuracy(y_true, obj.y_pre)

print(np.mean(acc), "time = ", t0, t1)


#  paper, Binalpha, acc = 0.421
#  run,   Binalpha, acc = 0.441
