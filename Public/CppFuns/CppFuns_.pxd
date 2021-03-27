from libcpp.vector cimport vector

cdef extern from "CppFuns.cpp":
    pass

cdef extern from "CppFuns.h" namespace "cf":
    void EuDist2(vector[vector[double]] &A, vector[double] &anorm, vector[vector[double]] &B, vector[double] &bnorm, vector[vector[double]] &C)
    void EuDist2_byCol(vector[vector[double]] &A, vector[double] &anorm, vector[vector[double]] &B, vector[double] &bnorm, vector[vector[double]] &C)
    void square_sum_by_row(vector[vector[double]] &X, vector[double] &norm);
    void square_sum_by_col(vector[vector[double]] &X, vector[double] &norm);
    void argsort_f(int *v, int n, int *ind)
    void symmetry(vector[vector[int]] &NN, vector[vector[double]] &NND, int argument)
    double maximum_2Dvec(vector[vector[double]] &Vec)

