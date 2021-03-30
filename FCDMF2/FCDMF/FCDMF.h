#ifndef FCDMF_H_
#define FCDMF_H_

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include "CppFuns.h"

using namespace std;

class FCDMF{
public:

    int c_true;
    vector<vector<double>> B;
    vector<vector<double>> BT;
    vector<int> y;

    FCDMF();
    FCDMF(vector<vector<double>> &B, int c_true);
    ~FCDMF();
    void opt(int ITER, vector<int> &p, vector<int> &q, int a1, int a2);

    void update_S(vector<int> &p, vector<int> &q, vector<vector<double>> &S);

    void compute_nc(vector<int> &y, vector<int> &ret);

    void update_pq(vector<vector<double>> &D, vector<int> &y, int L);

    void update_SQT(vector<vector<double>> &S, vector<int> &q, vector<vector<double>> &SQT);

    void update_STPT(vector<vector<double>> &S, vector<int> &p, vector<vector<double>> &STPT);

    double compute_obj(vector<vector<double>> &B, vector<int> &p, vector<int> &q, vector<vector<double>> &S);

};

#endif
