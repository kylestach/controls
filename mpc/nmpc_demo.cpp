#include "nmpc.h"
#include <iostream>

constexpr double dt = 0.05;

struct System {
    Vector<2> operator()(Vector<2> x, Vector<1> u, int t) {
        return Vector<2>(x(1), u(0) - 9.81 * std::sin(x(0)));
    }
};

struct Cost {
    double operator()(Vector<2> x, Vector<1> u, int t) {
        return 0.0 * u(0) * u(0);
    }
};

struct CostFinal {
    double operator()(Vector<2> x) {
        return 400 * (x(0) - 3.14) * (x(0) - 3.14) + 100 * x(1) * x(1);
    }
};

struct Constraint {
    Vector<1> operator()(Vector<2> x, Vector<1> u, int t) {
        return Vector<1>(0);
    }
};

int main() {
    System system;
    Cost cost;
    CostFinal cost_final;
    Constraint constraint;
    Discretized<2, 1, System> disc{dt, system};
    int N = 120;
    NonlinearModelPredictiveController<2, 1, 1, Discretized<2, 1, System>, Cost, CostFinal, Constraint> nmpc(
            disc, cost, cost_final, constraint,
            Vector<1>(-7), Vector<1>(7),
            Vector<1>(-2), Vector<1>(2),
            N + 1
        );

    std::cerr << "Factoring complete" << std::endl;

    Vector<2> x(0, 0);
    Vector<> u = nmpc.plan(x, Vector<2>::Zero());
    for (int i = 0; i < N; i++) {
        std::cout << x(0) << ", " << x(1) << ", " << u(i) << std::endl;
        x = disc(x, Vector<1>(u(i)), i);
    }
    std::cout << x(0) << ", " << x(1) << std::endl;
}
