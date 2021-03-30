#ifndef FCDMF_H_
#define FCDMF_H_

#include <iostream>
#include <valarray>
#include <chrono>
#include <valarray>
#include <algorithm>
#include "CppFuns.h"
#include <threads.h>

using namespace std;

class FCDMF{
public:

    int num;
    int num_anchor;
    int c_true;

    valarray<valarray<double>> B;
    valarray<valarray<double>> BT;
    vector<int> y;

    FCDMF();
    FCDMF(vector<vector<double>> &B, int c_true);
    ~FCDMF();
    void opt(int ITER, vector<int> &p, vector<int> &q, int a1, int a2);

    void update_S(valarray<int> &p, valarray<int> &q, valarray<valarray<double>> &S);

    void compute_nc(valarray<int> &y, valarray<int> &ret);

    void update_pq(valarray<valarray<double>> &D, valarray<int> &y, int L);

    void update_SQT(valarray<valarray<double>> &S, valarray<int> &q, valarray<valarray<double>> &SQT);

    void update_STPT(valarray<valarray<double>> &S, valarray<int> &p, valarray<valarray<double>> &STPT);

    double compute_obj(valarray<valarray<double>> &B, valarray<int> &p, valarray<int> &q, valarray<valarray<double>> &S);

};

#endif
