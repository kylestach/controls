#pragma once

#include "system.h"
#include <qpOASES.hpp>

template<int N, int M>
class ModelPredictiveController {
public:
    ModelPredictiveController(Matrix<N> A, Matrix<N, M> B, Vector<M> lb, Vector<M> ub, Matrix<N> Q, Matrix<M> R, int L)
            : problem(M * (L - 1)), A(A), B(B), Q(Q), R(R), L(L) {
        // Minimize sum(i = 0 to L - 1)(1/2[x^T Q x + u^T R u]) + x^T Qf x
        // Subject to x(i + 1) = Ax(i) + Bu(i)
        //
        // Make a big transition matrix:
        // x(0) = x(0)
        // x(1) = Ax(0) + Bu(0)
        // x(2) = A^2x(0) + ABu(0) + Bu(1)
        // x(3) = A^3x(0) + A^2Bu(0) + ABu(1) + Bu(2)
        // ...
        // x(L) = A^Lx(0) + A^(L-1)Bu(0) + A^(L-2)Bu(1)
        // So we have:
        // [x(0)]   = [ 0   0  ...  0  0 ] [ u(0) ]  + [ x(0)    ]
        // [x(1)]   = [ B   0  ...  0  0 ] [ u(1) ]  + [ Ax(0)   ]
        // [x(2)]   = [ AB  B  ...  0  0 ] [ u(2) ]  + [ A^2x(0) ]
        // [x(3)]   = [A^2B AB ...  0  0 ] [ u(2) ]  + [ A^3x(0) ]
        //   .      .   .   .  ...  .  .     .       .    .
        // [x(L)]   = [A^LB .  ...  B  0 ] [ u(L) ]  + [ A^Lx(0) ]
        // We can ignore the first row (you can't affect x0) and the last column (the last input does nothing)
        A_aug = Matrix<>::Zero(N * (L - 1), N * (L - 1));
        B_aug = Matrix<>::Zero(N * (L - 1), M * (L - 1));
        Q_aug = Matrix<>::Zero(N * (L - 1), N * (L - 1));
        R_aug = Matrix<>::Zero(M * (L - 1), M * (L - 1));
        lb_aug = Vector<>::Zero(M * (L - 1));
        ub_aug = Vector<>::Zero(M * (L - 1));

        Matrix<> A_i = Matrix<>::Identity(N, N);
        for (int i = 0; i < (L - 1); i++) {
            A_aug.template block<N, N>(N * i, N * i) = A_i;
            Q_aug.template block<N, N>(N * i, N * i) = Q;
            R_aug.template block<M, M>(M * i, M * i) = R;
            lb_aug.template block<M, 1>(M * i, 0) = lb;
            ub_aug.template block<M, 1>(M * i, 0) = ub;

            for (int j = 0; i + j < (L - 1); j++) {
                B_aug.template block<N, M>(N * (i + j), M * j) = A_i * B;
            }
            A_i *= A;
        }

        cost_quadratic = R_aug + B_aug.transpose() * Q_aug * B_aug;

        Matrix<> cost_linear = Matrix<>::Zero(1, M * (L - 1));

        qpOASES::Options options;
        options.printLevel = qpOASES::PL_NONE;
        problem.setOptions(options);

        int nWSR = 1000000;
        problem.init(cost_quadratic.data(), cost_linear.data(), lb_aug.data(), ub_aug.data(), nWSR);
    }

    Vector<M> control(Vector<N> x0, Vector<N> r) {
        Vector<> Ax_aug = Vector<>::Zero(N * (L - 1));
        Vector<> r_aug = Vector<>::Zero(N * (L - 1));

        Matrix<> A_i = Matrix<>::Identity(N, N);
        for (int i = 0; i < (L - 1); i++) {
            Ax_aug.template block<N, 1>(N * i, 0) = A_i * x0;
            r_aug.template block<N, 1>(N * i, 0) = r;
            A_i *= A;
        }

        // cost = 1/2[(x - r)^T Q (x - r) + u^T R u]
        // x = Ax0 + Bu
        // cost = 1/2[x0^T A^T Q A x0 - 2r^T Q (Ax0 + Bu) + 2x0^T A^T Q B u + u^T (R + B^T Q B) u + r^T Q r]
        // cost = 1/2 u^T [R + B^T Q B] u + [x0^T A^T Q B - r^T Q B] u
        Matrix<> cost_linear = Ax_aug.transpose() * Q_aug * B_aug - r_aug.transpose() * Q_aug * B_aug;

        int nWSR = 100000000;
        double cputime = 100000000;

        Vector<> u_min = Vector<>::Zero(M * (L - 1));
        problem.hotstart(cost_linear.data(), lb_aug.data(), ub_aug.data(), nWSR, &cputime);
        problem.getPrimalSolution(u_min.data());

        return u_min.template block<M, 1>(0, 0);
    }

protected:
    qpOASES::QProblemB problem;
    Matrix<N, N> A;
    Matrix<N, M> B;
    Vector<> lb_aug, ub_aug;
    Matrix<N, N> Q;
    Matrix<M, M> R;
    int L;

    Matrix<> A_aug;
    Matrix<> B_aug;
    Matrix<> Q_aug;
    Matrix<> R_aug;
    Matrix<> cost_quadratic;
};
