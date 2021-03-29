#include "FCDMF.h"

FCDMF::FCDMF(){}

FCDMF::FCDMF(vector<vector<double>> &B, int c_true){
    this->B = B;
    this->c_true = c_true;
    this->y = vector<int>(B.size(), 0);

    this->BT = vector<vector<double>>(B[0].size(), vector<double>(B.size(), 0));
    for (int i = 0; i < B.size(); i++){
        for (int j = 0; j < B[0].size(); j++){
            BT[j][i] = B[i][j];
        }
    }
}

FCDMF::~FCDMF(){}

void FCDMF::opt(int ITER, vector<int> &p, vector<int> &q, int a1, int a2){
    vector<double> bnorm(B.size(), 0);
    cf::square_sum_by_row(B, bnorm);

    vector<double> btnorm(BT.size(), 0);
    cf::square_sum_by_row(BT, btnorm);

    vector<vector<double>> S(c_true, vector<double>(c_true, 0));
    vector<vector<double>> SQT(c_true, vector<double>(B[0].size(), 0));
    vector<vector<double>> STPT(c_true, vector<double>(B.size(), 0));
    vector<vector<double>> DBP(B.size(), vector<double>(c_true, 0));
    vector<vector<double>> DBQ(B[0].size(), vector<double>(c_true, 0));

    vector<double> v(c_true, 0);
    vector<double> obj(ITER, 0);
    double tmp_d;

    // chrono::milliseconds eu1_t = chrono::milliseconds(0);
    // chrono::milliseconds eu2_t = chrono::milliseconds(0);
    // chrono::milliseconds pq_t = chrono::milliseconds(0);
    // chrono::milliseconds total_t = chrono::milliseconds(0);
    // chrono::time_point<chrono::steady_clock> t1;
    // chrono::time_point<chrono::steady_clock> t2;
    // chrono::time_point<chrono::steady_clock> t3;
    // chrono::time_point<chrono::steady_clock> t4;
    //
    // t3 = chrono::steady_clock::now();
    int Iter = 0;
    for (Iter = 0; Iter < ITER; Iter++){

        update_S(p, q, S);

        update_SQT(S, q, SQT);

        // t1 = chrono::steady_clock::now();
        cf::square_sum_by_row(SQT, v);
        cf::EuDist2(B, bnorm, SQT, v, DBP);
        // t2 = chrono::steady_clock::now();
        // eu1_t += chrono::duration_cast<chrono::milliseconds> (t2 - t1);

        // t1 = chrono::steady_clock::now();
        update_pq(DBP, p, a1);
        // t2 = chrono::steady_clock::now();
        // pq_t += chrono::duration_cast<chrono::milliseconds> (t2 - t1);

        update_STPT(S, p, STPT);

        // t1 = chrono::steady_clock::now();
        cf::square_sum_by_row(STPT, v);
        cf::EuDist2(BT, btnorm, STPT, v, DBQ);
        // t2 = chrono::steady_clock::now();
        // eu2_t += chrono::duration_cast<chrono::milliseconds> (t2 - t1);

        // t1 = chrono::steady_clock::now();
        update_pq(DBQ, q, a2);
        // t2 = chrono::steady_clock::now();
        // pq_t += chrono::duration_cast<chrono::milliseconds> (t2 - t1);

        obj[Iter] = compute_obj(B, p, q, S);
        if (Iter > 2 && (obj[Iter] - obj[Iter - 1])/obj[Iter-1] < 1e-6){
            break;
        }
    }
    // t4 = chrono::steady_clock::now();
    // total_t += chrono::duration_cast<chrono::milliseconds> (t4 - t3);
    y = p;
    // cout << "eu1 time = " << eu1_t.count() << endl;
    // cout << "eu2 time = " << eu2_t.count() << endl;
    // cout << "pq time = " << pq_t.count() << endl;
    // cout << "total time = " << total_t.count() << endl;
    // cout << "iter = " << Iter << endl;
}


void FCDMF::update_S(vector<int> &p, vector<int> &q, vector<vector<double>> &S){

    #pragma omp parallel for
    for (int i = 0; i < c_true; i++){
        fill(S[i].begin(), S[i].end(), 0);
    }
    
    for (int i = 0; i < B.size(); i++){
        for (int j = 0; j < B[0].size(); j++){
            S[p[i]][q[j]] += B[i][j];
        }
    }

    vector<int> pnc(c_true, 0);
    compute_nc(p, pnc);

    vector<int> qnc(c_true, 0);
    compute_nc(q, qnc);

    for (int i = 0; i < c_true; i++){
        for (int j = 0; j < c_true; j++){
            S[i][j] /= pnc[i]*qnc[j];
        }
    }

}

void FCDMF::update_pq(vector<vector<double>> &D, vector<int> &y, int L){
    vector<int> nc(c_true, 0);
    compute_nc(y, nc);

    int c_old, c_new, converge;
    while (2>1){
        converge = 1;
        for (int i = 0; i < y.size(); i++){
            c_old = y[i];
            if (nc[c_old] <= L){
                continue;
            }

            c_new = distance(D[i].begin(), min_element(D[i].begin(), D[i].end()));

            if (c_new != c_old){
                y[i] = c_new;
                nc[c_old] -= 1;
                nc[c_new] += 1;
                converge = 0;
            }
        }

        if (converge == 1){
            break;
        }
    }
}

void FCDMF::compute_nc(vector<int> &y, vector<int> &ret){
    fill(ret.begin(), ret.end(), 0);
    for (int i = 0; i < y.size(); i++){
        ret[y[i]] += 1;
    }
}

void FCDMF::update_SQT(vector<vector<double>> &S, vector<int> &q, vector<vector<double>> &SQT){
    int tmp_c = 0;
    for (int i = 0; i < SQT[0].size(); i++){
        tmp_c = q[i];
        for (int j = 0; j < SQT.size(); j++){
            SQT[j][i] = S[j][tmp_c];
        }
    }
}

void FCDMF::update_STPT(vector<vector<double>> &S, vector<int> &p, vector<vector<double>> &STPT){
    int tmp_c = 0;
    for (int i = 0; i < STPT[0].size(); i++){
        tmp_c = p[i];
        for (int j = 0; j < STPT.size(); j++){
            STPT[j][i] = S[tmp_c][j];
        }
    }
}

double FCDMF::compute_obj(vector<vector<double>> &B, vector<int> &p, vector<int> &q, vector<vector<double>> &S){
    double obj = 0;
    double tmp_d = 0;
    for (int i = 0; i < B.size(); i++){
        for (int j = 0; j < B[0].size(); j++){
            tmp_d = B[i][j] - S[p[i]][q[j]];
            obj += tmp_d * tmp_d;
        }
    }
    return obj;
}

