#pragma once

#include "eigen3/Eigen/Core"

template<int N = Eigen::Dynamic, int M = N>
using Matrix = Eigen::Matrix<double, N, M, M == 1 ? Eigen::ColMajor : Eigen::RowMajor>;

template<int N = Eigen::Dynamic>
using Vector = Eigen::Matrix<double, N, 1>;

template<int N, int M, typename T>
struct Discretized {
    double dt;
    T system;

    Vector<N> operator()(Vector<N> x, Vector<M> u, double t) {
        Vector<N> k1 = dt * system(x, u, t);
        Vector<N> k2 = dt * system(Vector<N>(x + k1 / 2), u, t + dt / 2);
        Vector<N> k3 = dt * system(Vector<N>(x + k2 / 2), u, t + dt / 2);
        Vector<N> k4 = dt * system(Vector<N>(x + k3), u, t + dt);
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }
};

template<int N, int M, typename T>
Discretized<N, M, T> discretize(T sys, double dt) {
    return {dt, sys};
}

template<int N, int M, int P = N>
struct Linearized {
    Matrix<P, N> A;
    Matrix<P, M> B;
    Matrix<P, 1> c;
};

template<int N, int M, typename System, int P = N>
Linearized<N, M, P> linearize(System sys, Vector<N> x, Vector<M> u, int t) {
    Linearized<N, M, P> result;

    Vector<P> value = sys(x, u, t);

    const double epsx = 1e-8 * std::max(1e-2, x.norm());

    for (int i = 0; i < N; i++) {
        Vector<N> dx = Vector<N>::Zero();
        dx(i) = epsx;
        result.A.template block<P, 1>(0, i) = (sys(x + dx, u, t) - value) / epsx;
    }

    const double epsu = 1e-8 * std::max(1e-2, u.norm());
    for (int i = 0; i < M; i++) {
        Vector<M> du = Vector<M>::Zero();
        du(i) = epsu;
        result.B.template block<P, 1>(0, i) = (sys(x, u + du, t) - value) / epsu;
    }

    result.c = value - result.A * x - result.B * u;

    return result;
}

template<int N, int M>
struct Quadratized {
    Matrix<N> Q;
    Matrix<N, M> P;
    Matrix<M> R;
    Matrix<N, 1> q;
    Matrix<M, 1> r;
    double c;
};

template<int N, int M, typename Cost>
Quadratized<N, M> quadratize(Cost cost, Vector<N> x, Vector<M> u, int t) {
    Quadratized<N, M> result;

    double value = cost(x, u, t);

    constexpr double eps = 1e-5;

    for (int i = 0; i < N; i++) {
        Vector<N> dx1 = Vector<N>::Zero();
        dx1(i) = eps;
        double dc_dx1 = (cost(x + dx1, u, t) - value) / eps;
        for (int j = i; j < N; j++) {
            Vector<N> dx2 = Vector<N>::Zero();
            dx2(j) = eps;
            double dc_dx1_stepped = (cost(x + dx1 + dx2, u, t) - cost(x + dx2, u, t)) / eps;
            result.Q(i, j) = result.Q(j, i) = (dc_dx1_stepped - dc_dx1) / eps;
        }
        for (int j = 0; j < M; j++) {
            Vector<M> du1 = Vector<M>::Zero();
            du1(j) = eps;
            double dc_dx1_stepped = (cost(x + dx1, u + du1, t) - cost(x, u + du1, t)) / eps;
            result.P(i, j) = (dc_dx1_stepped - dc_dx1) / eps;
        }
        result.q(i) = dc_dx1;
    }

    for (int i = 0; i < M; i++) {
        Vector<M> du1 = Vector<M>::Zero();
        du1(i) = eps;
        double dc_du1 = (cost(x, u + du1, t) - value) / eps;
        for (int j = i; j < M; j++) {
            Vector<M> du2 = Vector<M>::Zero();
            du2(j) = eps;
            double dc_du1_stepped = (cost(x, u + du1 + du2, t) - cost(x, u + du2, t)) / eps;
            result.R(i, j) = result.R(j, i) = (dc_du1_stepped - dc_du1) / eps;
        }
        result.r(i) = dc_du1;
    }

    result.q -= result.Q * x + result.P * u;
    result.r -= result.R * u + x.transpose() * result.P;

    result.c = value - (
            x.dot(0.5 * result.Q * x + result.P * u + result.q) +
            u.dot(0.5 * result.R * u + result.r));

    return result;
}

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
