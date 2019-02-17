#pragma once

#include "common/system.h"

template<int M, int N, int L, typename System>
class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter(System sys, GaussianNoise<N, L> noise, Vector<N> x0, Matrix<N> P0)
        : sys(sys), noise(noise), x(x0), u_prev(Vector<M>::Zero()), P(P0) {}

    void predict(Vector<M> u, size_t t = 0) {
        auto l = linearize<N, M, System>(sys, x, u, t);
        x = l.A * x + l.B * u + l.c;
        P = l.A * P * l.A.transpose() + noise.Q;
    }

    void update(Vector<L> y, size_t t = 0) {
        auto ls = linearize<N, M, System>(sys, x, u_prev, t);
        Vector<L> y_err = y - sys.measure(x, u_prev, t);
        Matrix<L> S = noise.R + ls.C * P * ls.C.transpose();
        Matrix<N, L> K = P * ls.C.transpose() * S.inverse();
        x += K * y_err;
        Matrix<N> IsubKC = Matrix<N>::Identity() - K * ls.C;
        P = IsubKC * P * IsubKC.transpose() + K * noise.R * K.T;
    }

private:
    System sys;
    GaussianNoise<N, L> noise;
    Vector<N> x;
    Vector<M> u_prev;
    Matrix<N> P;
};
