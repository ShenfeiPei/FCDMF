cimport numpy as np
import numpy as np
np.import_array()
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

from cython cimport view

from .FCDMF_ cimport FCDMF

from FCDMF_pack.Public import Ifuns

cdef class PyFCDMF():
    cdef FCDMF c_FCDMF
    cdef int num
    cdef int num_anchor
    cdef int c_true
    cdef double[:, :] B
    cdef int[:, :] Y

    def __init__(self, np.ndarray[double, ndim=2] B, int c_true):
        self.B = B
        self.c_FCDMF = FCDMF(B, c_true)
        self.num = B.shape[0]
        self.num_anchor = B.shape[1]
        self.c_true = c_true

    def opt(self, int rep, int ITER, init="k-means++"):
        #  self.Y = view.array(shape=(rep, self.num), itemsize=sizeof(int), format="i")
        Y = np.zeros((rep, self.num), dtype=np.int32)
        a1 = int(np.maximum(self.num/10/self.c_true, 1))
        a2 = int(np.maximum(self.num_anchor/10/self.c_true, 1))

        B_sp = sparse.csr_matrix(self.B)

        if init=="random":
            P = Ifuns.initialY(init, self.num, self.c_true, rep=rep)
            Q = Ifuns.initialY(init, self.num_anchor, self.c_true, rep=rep)
        else:
            U, S, VH = svds(B_sp, k=self.c_true, which="LM")
            P = Ifuns.initialY(init, self.num, self.c_true, rep=rep, X=U[:, :self.c_true])
            V = VH.T
            Q = Ifuns.initialY(init, self.num_anchor, self.c_true, rep=rep, X=V[:, :self.c_true])

        for rep_i in range(rep):

            p = P[rep_i, :]
            q = Q[rep_i, :]
            self.c_FCDMF.opt(ITER, p, q, a1, a2)

            Y[rep_i, :] = np.array(self.c_FCDMF.y)

        self.Y = Y

    @property
    def y_pre(self):
        return np.array(self.Y)

    @property
    def ref(self):
        titile = "Fast Clustering with Co-Clustering via Discrete Non-negative matrix factorization for image identification, ICASSP, 2020"

