#include "FCDMF.h"

FCDMF::FCDMF(){}

FCDMF::FCDMF(vector<vector<double>> &B, int c_true){
    this->num = B.size();
    this->num_anchor = B[0].size();

    this->B.resize(num);
    for (int i = 0; i < num; i++){
        this->B[i].resize(num_anchor);
    }
    for (int i = 0; i < num; i++){
        for (int j = 0; j < num_anchor; j++){
            this->B[i][j] = B[i][j];
        }
    }

    this->c_true = c_true;
    this->y.resize(B.size());

    this->BT.resize(num_anchor);
    for (int i = 0; i < num_anchor; i++){
        this->BT[i].resize(num);
    }
    for (int i = 0; i < num; i++){
        for (int j = 0; j < num_anchor; j++){
            this->BT[j][i] = B[i][j];
        }
    }
}

FCDMF::~FCDMF(){}

void FCDMF::opt(int ITER, vector<int> &p2, vector<int> &q2, int a1, int a2){
    valarray<int> p(p2.size());
    valarray<int> q(q2.size());
    for (int i = 0; i < p.size(); i++){
        p[i] = p2[i];
    }
    for (int i = 0; i < q.size(); i++){
        q[i] = q2[i];
    }
    
    // for (int i = 0; i < 5; i++){
    //     for (int j = 0; j < 5; j++){
    //         cout << B[i][j] << ", ";
    //     }
    //     cout << endl;
    // }
    //
    // for (int i = 0; i < 5; i++){
    //     for (int j = 0; j < 5; j++){
    //         cout << BT[i][j] << ", ";
        // }
        // cout << endl;
    // }
    valarray<double> bnorm(B.size());
    // cout << "bnorm size = " << bnorm.size() << endl;
    cf::square_sum_by_row(B, bnorm);

    valarray<double> btnorm(BT.size());
    // cout << "btnorm size = " << btnorm.size() << endl;
    cf::square_sum_by_row(BT, btnorm);

    valarray<valarray<double>> S;
    S.resize(c_true);
    for (int i = 0; i < S.size(); i++){
        S[i].resize(c_true);
    }

    valarray<valarray<double>> SQT;
    SQT.resize(c_true);
    for (int i = 0; i < c_true; i++){
        SQT[i].resize(num_anchor);
    }

    valarray<valarray<double>> STPT;
    STPT.resize(c_true);
    for (int i = 0; i < c_true; i++){
        STPT[i].resize(num);
    }

    valarray<valarray<double>> DBP;
    DBP.resize(num);
    for (int i = 0; i < num; i++){
        DBP[i].resize(c_true);
    }
    valarray<valarray<double>> DBQ;
    DBQ.resize(num_anchor);
    for (int i = 0; i < num_anchor; i++){
        DBQ[i].resize(c_true);
    }

    valarray<double> v(c_true);
    valarray<double> obj(ITER);
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
    for (int i = 0; i < y.size(); i++){
        y[i] = p[i];
    }
    // cout << "y size = " << y.size() << endl;
    // cout << "p size = " << p.size() << endl;
    // cout << "eu1 time = " << eu1_t.count() << endl;
    // cout << "eu2 time = " << eu2_t.count() << endl;
    // cout << "pq time = " << pq_t.count() << endl;
    // cout << "total time = " << total_t.count() << endl;
    // cout << "Iter = " << Iter << endl;
}


void FCDMF::update_S(valarray<int> &p, valarray<int> &q, valarray<valarray<double>> &S){

    #pragma omp parallel for
    for (int i = 0; i < c_true; i++){
        // fill(S[i].begin(), S[i].end(), 0);
        S[i] = 0;
    }

    for (int i = 0; i < B.size(); i++){
        for (int j = 0; j < B[0].size(); j++){
            S[p[i]][q[j]] += B[i][j];
        }
    }

    valarray<int> pnc(c_true);
    compute_nc(p, pnc);

    valarray<int> qnc(c_true);
    compute_nc(q, qnc);

    for (int i = 0; i < c_true; i++){
        for (int j = 0; j < c_true; j++){
            S[i][j] /= pnc[i]*qnc[j];
        }
    }

}

void FCDMF::update_pq(valarray<valarray<double>> &D, valarray<int> &y, int L){
    valarray<int> nc(c_true);
    compute_nc(y, nc);

    int c_old, c_new, converge;
    while (2>1){
        converge = 1;
        for (int i = 0; i < y.size(); i++){
            c_old = y[i];
            if (nc[c_old] <= L){
                continue;
            }
            c_new = distance(begin(D[i]), min_element(begin(D[i]), end(D[i])));

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

void FCDMF::compute_nc(valarray<int> &y, valarray<int> &ret){
    ret = 0;
    // fill(ret.begin(), ret.end(), 0);
    for (int i = 0; i < y.size(); i++){
        ret[y[i]] += 1;
    }
}

void FCDMF::update_SQT(valarray<valarray<double>> &S, valarray<int> &q, valarray<valarray<double>> &SQT){
    int tmp_c = 0;
    for (int i = 0; i < SQT[0].size(); i++){
        tmp_c = q[i];
        for (int j = 0; j < SQT.size(); j++){
            SQT[j][i] = S[j][tmp_c];
        }
    }
}

void FCDMF::update_STPT(valarray<valarray<double>> &S, valarray<int> &p, valarray<valarray<double>> &STPT){
    int tmp_c = 0;
    for (int i = 0; i < STPT[0].size(); i++){
        tmp_c = p[i];
        for (int j = 0; j < STPT.size(); j++){
            STPT[j][i] = S[tmp_c][j];
        }
    }
}

double FCDMF::compute_obj(valarray<valarray<double>> &B, valarray<int> &p, valarray<int> &q, valarray<valarray<double>> &S){
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

