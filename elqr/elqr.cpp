#include <iostream>
#include <vector>
#include <fstream>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "common/system.h"

template<int N, int M, typename System, typename Cost, typename CostFinal>
void elqr(System sys, Cost cost, CostFinal cost_final, Vector<N> x0, int L) {
    constexpr int num_iterations = 10;

    std::vector<Matrix<N>> S(L + 1), Sbar(L + 1);
    std::vector<Vector<N>> s(L + 1), sbar(L + 1);
    std::vector<Matrix<M, N>> K(L), Kbar(L);
    std::vector<Matrix<M, 1>> k(L), kbar(L);

    for (int i = 0; i <= L; i++) {
        S[i] = Sbar[i] = Matrix<N>::Zero();
        s[i] = sbar[i] = Vector<N>::Zero();
        if (i < L) {
            K[i] = Kbar[i] = Matrix<M, N>::Zero();
            k[i] = kbar[i] = Vector<M>::Zero();
        }
    }

    constexpr double dt = 0.01;
    Discretized<N, M, System> g{dt, sys};
    Discretized<N, M, System> gbar{-dt, sys};

    for (int i = 0; i < num_iterations; i++) {
        // Forwards pass
        for (int t = 0; t < L; t++) {
            Vector<N> x_prev;
            if (i == 0 && t == 0) {
                x_prev = x0;
            } else {
                x_prev = -(S[t] + Sbar[t]) * (s[t] + sbar[t]);
            }
            Vector<M> u = K[t] * x_prev + kbar[t];
            Vector<N> x = g(x_prev, u, t);

            Linearized<N, M> l = linearize(gbar, x, u, t);
            Quadratized<N, M> q = quadratize(cost, x, u, t);

            Matrix<N> Sbar_prev = Sbar[t];
            Vector<N> sbar_prev = sbar[t];

            Matrix<M, N> Cbar = l.B.transpose() * (Sbar_prev + q.Q) * l.A +
                                q.P.transpose() * l.A;
            Matrix<N> Dbar = l.A.transpose() * (Sbar_prev + q.Q) * l.A;
            Matrix<M> Ebar =
                    l.B.transpose() * (Sbar_prev + q.Q) * l.B + q.R
                    + q.P.transpose() * l.B + l.B.transpose() * q.P;
            Vector<N> dbar =
                    l.A.transpose() * (sbar_prev + q.q) + l.A.transpose() * (Sbar_prev + q.Q) * l.c;
            Vector<M> ebar =
                    q.r + q.P.transpose() * l.c + l.B.transpose() * (sbar_prev + q.q) +
                    l.B.transpose() * (Sbar_prev + q.Q) * l.c;

            Matrix<M> Ebar_inv = Ebar.inverse();
            Sbar[t + 1] = Dbar - Cbar.transpose() * Ebar_inv * Cbar;
            sbar[t + 1] = dbar - Cbar.transpose() * Ebar_inv * ebar;
            Kbar[t] = -Ebar_inv * Cbar;
            kbar[t] = -Ebar_inv * ebar;
        }

        // Backwards pass
        Vector<N> xL;
        xL = -(S[L] + Sbar[L]).inverse() * (s[L] + sbar[L]);
        QuadratizedFinal<N> final_quadratized = quadratize_final(cost_final, xL);

        S[L] = final_quadratized.Q;
        s[L] = final_quadratized.q;
        for (int t = L - 1; t >= 0; t--) {
            Vector<N> x_next = -(S[t + 1] + Sbar[t + 1]) * (s[t + 1] + sbar[t + 1]);
            Vector<M> u = Kbar[t] * x_next + kbar[t];
            Vector<N> x = gbar(x_next, u, t);

            Linearized<N, M> l = linearize(g, x, u, t);
            Quadratized<N, M> q = quadratize(cost, x, u, t);

            Matrix<N> S_next = S[t + 1];
            Vector<N> s_next = s[t + 1];

            Matrix<M, N> C = q.P.transpose() + l.B.transpose() * S_next * l.A;
            Matrix<N> D = q.Q + l.A.transpose() * S_next * l.A;
            Matrix<M> E = q.R + l.B.transpose() * S_next * l.B;
            Vector<N> d = q.q + l.A.transpose() * (s_next + S_next * l.c);
            Vector<M> e = q.r + l.B.transpose() * (s_next + S_next * l.c);

            Matrix<M> E_inv = E.inverse();
            S[t] = D - C.transpose() * E_inv * C;
            s[t] = d - C.transpose() * E_inv * e;

            K[t] = -E_inv * C;
            k[t] = -E_inv * e;
        }
    }

    std::ofstream file("/tmp/test.csv");
    for (int t = 0; t <= L; t++) {
        Vector<2> x = -(S[t] + Sbar[t]).inverse() * (s[t] + sbar[t]);
        file << x(0) << "\t" << x(1) << "\n";
    }
}

struct ContinuousPendulum {
    Vector<2> operator()(Vector<2> x, Vector<1> u, int t) {
        return Vector<2>(x(1), u(0) - 9.81 * std::sin(x(0)));
    }
};

struct ExponentialCost {
    double operator()(Vector<2> x, Vector<1> u, int t) {
        return x(0) * x(0) + u(0) * u(0);
    }
};

struct ExponentialFinalCost {
    double operator()(Vector<2> x) {
        return 1000 * (x(0) - 3.14159) * (x(0) - 3.14159) + 400 * x(1) * x(1);
    }
};

template<>
Quadratized<2, 1> quadratize(ExponentialCost cost, Vector<2> x, Vector<1> u, int t) {
    Quadratized<2, 1> result;
    result.Q << 0.0, 0.0,
            0.0, 0.0;
    if (t == 0) {
        result.Q(0, 0) = 2000.0;
        result.Q(1, 1) = 2000.0;
    }
    result.P << 0.0,
            0.0;
    result.R << 0.1;
    result.q << 0.0, 0.0;
    result.r << 0.0;
    result.c = 0.0;
    return result;
}

template<>
Linearized<2, 1> linearize(Discretized<2, 1, ContinuousPendulum> sys, Vector<2> x, Vector<1> u, int t) {
    Linearized<2, 1> result;
    result.A << 1, sys.dt,
            -9.81 * sys.dt * (std::cos(x(0)) + 1.5), 0.995;
//    result.A << 1, sys.dt, -9.81 * sys.dt, 1;
    result.B << 0.5 * sys.dt * sys.dt,
            sys.dt;
    result.c << 0.0, 0.0;
    return result;
}

int main() {
    ExponentialCost e;
    ExponentialFinalCost ef;
    ContinuousPendulum p;
    Vector<2> x(1, 3);

    elqr<2, 1, ContinuousPendulum, ExponentialCost, ExponentialFinalCost>(p, e, ef, x, 500);
}
