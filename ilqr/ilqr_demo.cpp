#include "ilqr.h"

#include <vector>
#include <limits>
#include <iostream>

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

    int N = 600;
    Policy<2, 1> p = ilqr<2, 1, ContinuousPendulum, QuadraticCost, ExponentialFinalCost>(pendulum, cost, cost_final, x, Vector<2>(M_PI, 0), N);
    for (int t = 0; t < N; t++) {
        Vector<1> u = -(p.K[t] * x + p.k[t]);
        std::cout << x(0) << ", " << x(1) << ", " << u(0) << std::endl;
        x = g(x, u, t);
    }
}
