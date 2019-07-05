#include "ekf.hpp"
#include "eigen3/Eigen/Dense"

constexpr double dt = 0.1;

struct System {
    Vector<2> operator()(Vector<2> x, Vector<1> u, int) {
        return Vector<2>(x(1), u(0) - 9.81 * std::sin(x(0)));
    }

    Vector<1> measure(Vector<2> x, Vector<1> = Vector<1>::Zero(), int = 0) {
        return Vector<1>(x(0));
    }
};

int main() {
    System sys;
    GaussianNoise<2, 1> noise {Matrix<2>::Identity(), Matrix<1>::Identity()};
    UnscentedKalmanFilter<1, 2, 1, System>(
            sys, noise, Vector<2>(0, 0), Matrix<2>::Identity());
}
