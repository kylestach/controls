#pragma once

#include "common/system.h"
#include "eigen3/Eigen/Dense"

template<int N, int M>
void lqr_step(
        Linearized<N, M> l,
        Quadratized<N, M> q,
        Matrix<N, N> S_tn,
        Matrix<N, 1> s_tn,
        Matrix<N, N>& S_t,
        Matrix<N, 1>& s_t,
        Matrix<M, N>& K_t,
        Matrix<M, 1>& k_t) {
    Matrix<N, M> C = q.P + l.A.transpose() * S_tn * l.B;
    Matrix<N, N> D = q.Q + l.A.transpose() * S_tn * l.A;
    Matrix<M, M> E = q.R + l.B.transpose() * S_tn * l.B;
    Matrix<N, 1> d = q.q + l.A.transpose() * (s_tn + S_tn * l.c);
    Matrix<M, 1> e = q.r + l.B.transpose() * (s_tn + S_tn * l.c);

    Matrix<M, M> E_inv = E.inverse();
    S_t = D - C * E_inv * C.transpose();
    s_t = d - C * E_inv * e;
    K_t = E_inv * C.transpose();
    k_t = E_inv * e;
}

template<int N, int M, typename System, typename Cost, typename CostFinal>
Policy<N, M> ilqr(System sys, Cost cost, CostFinal cost_final, Vector<N> x0, Vector<N> xf, int L) {
    std::vector<Matrix<N>> S(L + 1);
    std::vector<Matrix<N, 1>> s(L + 1);

    std::vector<Matrix<M, N>> K(L);
    std::vector<Matrix<M, 1>> k(L);

    std::vector<Matrix<N, 1>> x(L + 1);

    constexpr double dt = 0.01;
    Discretized<N, M, System> g{dt, sys};

    for (int i = 0; i <= L; i++) {
        S[i] = Matrix<N>::Zero();
        s[i] = Matrix<N, 1>::Zero();
        double t = ((double) i) / L;
        x[i] = x0 + t * (xf - x0);
        if (i < L) {
            K[i] = Matrix<M, N>::Zero();
            k[i] = Matrix<M, 1>::Zero();
        }
    }

    double last_cost = std::numeric_limits<double>::infinity();
    constexpr int num_iterations = 100;
    for (int i = 0; i < num_iterations; i++) {
        QuadratizedFinal<N> qf = quadratize_final(cost_final, x[L]);
        S[L] = qf.Q;
        s[L] = qf.q;

        // Backwards pass: calculate S and K
        for (int t = L - 1; t >= 0; t--) {
            Matrix<M, 1> u = -(K[t] * x[t] + k[t]);
            Linearized<N, M> l = linearize(g, x[t], u, t);
            Quadratized<N, M> q = quadratize(cost, x[t], u, t);
            lqr_step(l, q, S[t + 1], s[t + 1], S[t], s[t], K[t], k[t]);
        }

        // Forwards pass: calculate x
        double total_cost = 0;
        for (int t = 0; t < L; t++) {
            Vector<M> u = -(K[t] * x[t] + k[t]);
            x[t + 1] = g(x[t], u, t);
            total_cost += cost(x[t], u, t);
        }
        total_cost += cost_final(x[L]);

        double delta_cost = last_cost - total_cost;
        double relative_cost = delta_cost / total_cost;
        if (std::abs(last_cost - total_cost) / total_cost < 1e-4) {
            break;
        }
        last_cost = total_cost;
    }

    return {std::move(K), std::move(k)};
}
