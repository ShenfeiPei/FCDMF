from libcpp.vector cimport vector

cdef extern from "CppFuns.cpp":
    pass

cdef extern from "FCDMF.cpp":
    pass

cdef extern from "FCDMF.h":
    cdef cppclass FCDMF:
        int num
        int anchor_num
        int c_true

        vector[vector[double]] B
        vector[int] y

        FCDMF() except+
        FCDMF(vector[vector[double]] &B, int c_true) except+
        void opt(int ITER, vector[int] &p, vector[int] &q, int a1, int a2)
        #  void update_S(vector[int] &p, vector[int] &q, vector[vector[double]] &S)
        #  void compute_nc(vector[int] &y, vector[int] &ret)
        #  void update_pq(vector[vector[double]] &D, vector[int] &y, int L)

