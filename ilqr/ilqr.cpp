#include "system.h"
#include "eigen3/Eigen/Dense"
#include <vector>
#include <limits>
#include <iostream>

template<int N>
struct QuadratizedFinal {
    Matrix<N> Q;
    Matrix<N, 1> q;
    double c;
};

template<int N, typename CostFinal>
QuadratizedFinal<N> quadratize_final(CostFinal cost, Vector<N> x) {
    QuadratizedFinal<N> result;

    Quadratized<N, 1> quad_full = quadratize([&cost](Vector<N> x, Vector<1>, int t) {
        return cost(x);
    }, x, Vector<1>(0), 0);

    result.Q = quad_full.Q;
    result.q = quad_full.q;
    result.c = quad_full.c;
    return result;
}

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

template<int N, int M>
struct Policy {
    std::vector<Matrix<M, N>> K;
    std::vector<Matrix<M, 1>> k;
};

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

struct ContinuousPendulum {
    Vector<2> operator()(Vector<2> x, Vector<1> u, int t) {
        return Vector<2>(x(1), u(0) - 9.81 * std::sin(x(0)));
    }
};

struct QuadraticCost {
    double operator()(Vector<2> x, Vector<1> u, int t) {
        return 1 * (u(0) * u(0));
    }
};

struct ExponentialFinalCost {
    double operator()(Vector<2> x) {
        return 4000 * (x(0) - M_PI) * (x(0) - M_PI) + 10000 * x(1) * x(1);
    }
};

int main() {
    ContinuousPendulum pendulum;
    QuadraticCost cost;
    ExponentialFinalCost cost_final;
    Discretized<2, 1, ContinuousPendulum> g{.01, pendulum};

    Vector<2> x(0, 0);

    Policy<2, 1> p = ilqr<2, 1, ContinuousPendulum, QuadraticCost, ExponentialFinalCost>(pendulum, cost, cost_final, x, Vector<2>(M_PI, 0), 500);
    for (int t = 0; t < 500; t++) {
        Vector<1> u = -(p.K[t] * x + p.k[t]);
        std::cout << x(0) << ", " << x(1) << ", " << u(0) << std::endl;
        x = g(x, u, t);
    }
}
