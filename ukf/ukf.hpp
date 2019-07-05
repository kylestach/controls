#pragma once

#include "common/system.h"
#include "eigen3/unsupported/Eigen/MatrixFunctions"

template<int M, int N, int L, typename System>
class UnscentedKalmanFilter {
public:
    UnscentedKalmanFilter(System sys, GaussianNoise<N, L> noise, Vector<N> x0, Matrix<N> P0)
        : sys(sys), noise(noise), x(x0), u_prev(Vector<M>::Zero()), P(P0) {}

    void predict(Vector<M> u, size_t t = 0) {
        Vector<2 * N> x_aug = Vector<2 * N>::Zero();
        x_aug.template block<N, 1>(0, 0) = x;
        Matrix<2 * N> P_aug = Matrix<2 * N>::Zero();
        P_aug.template block<N, N>(0, 0) = P;
        P_aug.template block<N, N>(N, N) = noise.Q;

        double alpha = 1e-3, beta = 2, kappa = 0;
        double lambda = alpha * alpha * (N + kappa) - N;
        Matrix<2 * N> sqrt_term = ((N + lambda) * P_aug).sqrt();

        x = Vector<N>::Zero();
        P = Matrix<N>::Zero();

        for (int i = 0; i < 2 * N; i++) {

        }

        Matrix<2 * N, 2 * N + 1>
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
