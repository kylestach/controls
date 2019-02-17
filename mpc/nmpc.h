#pragma once

#include <qpOASES.hpp>
#include "common/system.h"

template<int N, int M, int C, typename System, typename Cost, typename CostFinal, typename Constraint>
class NonlinearModelPredictiveController {
public:
    NonlinearModelPredictiveController(
            System sys, Cost cost, CostFinal cost_final, Constraint constraint, Vector<C> lb_constraint, Vector<C> ub_constraint, Vector<M> lb_input, Vector<M> ub_input, int L)
            : problem(M * (L - 1), C * (L - 1)),
              system(sys),
              constraint(constraint),
              cost(cost),
              cost_final(cost_final),
              lb_constraint(lb_constraint),
              ub_constraint(ub_constraint),
              lb_input(lb_input),
              ub_input(ub_input),
              L(L) {
        // Minimize sum(i = 0 to L - 1)(1/2[x^T Q x + u^T R u]) + x^T Qf x
        // Subject to x(i + 1) = Ax(i) + Bu(i)
        //
        // Make a big transition matrix:
        // x(0) = x(0)
        // x(1) = [B(0)u(0)] + [A(0)x(0) + c(0)]
        // x(2) = [A(1)B(0)u(0) + B(1)u(1)] + [A(1)A(0)x(0) + c(1) + A(1)c(0) = A(1)x(0) + c(1)]
        // x(3) = [A(2)A(1)B(0)u(0) + A(2)B(1)u(1)] + [A(2)A(1)A(0)x(0) + A(2)c(1) + A(2)A(1)c(0) = A(2)x(1) + c(2)]
        // ...
        // So we have (omitting the first row and the last column):
        // [x(0)]   = [ 0   0  ...  0  0 ] [ u(0) ]  + [ x(0)    ]
        // [x(1)]   = [ B   0  ...  0  0 ] [ u(1) ]  + [ Ax(0) + Ic ]
        // [x(2)]   = [ AB  B  ...  0  0 ] [ u(2) ]  + [ A^2x(0) + (A + I)c]
        // [x(3)]   = [A^2B AB ...  0  0 ] [ u(2) ]  + [ A^3x(0) + (A^2 + A + I)c]
        //   .      .   .   .  ...  .  .     .       .    .
        // [x(L)]   = [A^LB .  ...  B  0 ] [ u(L) ]  + [ A^Lx(0) + (A^(L-1) + ... + I)c]
        // We can ignore the first row (you can't affect x0) and the last column (the last input does nothing)
        //
        // Just like in regular linear MPC, we need to make a big transition matrix. However, at each point in
        // time we need to linearize around some x. We'll use the zero vector to initialize.
        //
        // We also have a constraint matrix. Our constraint is: lb_constraint <= H_u * u + H_x * x <= ub_constraint:
        // H_u * u + H_x * x + c = H_u * u + H_x * (x_aug + B_aug * u) + c = (H_u + H_x * B_aug)u + (H_x * x_aug + c)
        // So subtracting (H_x * x_aug + c) from the constraints, we get:
        // lb_constraint - H_x * x_aug - c <= (H_u + H_x * B_aug)u <= ub_constraint - H_x * x_aug - c

        Matrix<> B_aug = Matrix<>::Zero(N * (L - 1), M * (L - 1));
        Matrix<> x_aug = Vector<>::Zero(N * (L - 1));

        Matrix<> Q_aug = Matrix<>::Zero(N * (L - 1), N * (L - 1));
        Matrix<> R_aug = Matrix<>::Zero(M * (L - 1), M * (L - 1));
        Matrix<> q_aug = Matrix<>::Zero(1, N * (L - 1));
        Matrix<> r_aug = Matrix<>::Zero(1, N * (L - 1));

        Matrix<> lb_input_aug = Vector<>::Zero(M * (L - 1));
        Matrix<> ub_input_aug = Vector<>::Zero(M * (L - 1));
        Matrix<> lb_constraint_aug = Vector<>::Zero(C * (L - 1));
        Matrix<> ub_constraint_aug = Vector<>::Zero(C * (L - 1));

        Matrix<N> A_i = Matrix<N>::Identity();

        Matrix<> B_aug_row = Matrix<>::Zero(N, M * (L - 1));

        Matrix<> H_x = Matrix<>::Zero(C * (L - 1), N * (L - 1));
        Matrix<> H_u = Matrix<>::Zero(C * (L - 1), M * (L - 1));
        Matrix<> h_const = Matrix<>::Zero(C * (L - 1), 1);

        Vector<N> unforced_x = Vector<N>::Zero();

        Vector<N> x0 = Vector<N>::Zero();
        Vector<M> u0 = Vector<M>::Zero();
        for (int i = 0; i < (L - 1); i++) {
            auto linearized = linearize<N, M, System>(system, x0, u0, i);
            B_aug_row = linearized.A * B_aug_row;
            B_aug_row.template block<N, M>(0, M * i) = linearized.B;
            unforced_x = linearized.A * unforced_x + linearized.c;
            x_aug.template block<N, 1>(N * i, 0) = unforced_x;

            auto quadratized = quadratize<N, M, Cost>(cost, x0, u0, i);

            if (i != L - 2) {
                Q_aug.template block<N, N>(N * i, N * i) = quadratized.Q;
                q_aug.template block<1, N>(0, N * i) = quadratized.q;
            }

            R_aug.template block<M, M>(M * i, M * i) = quadratized.R;
            r_aug.template block<1, M>(0, M * i) = quadratized.r;

            auto constraint_linear = linearize<N, M, Constraint, C>(constraint, x0, u0, i);
            H_x.template block<C, N>(C * i, N * i) = constraint_linear.A;
            H_u.template block<C, M>(C * i, M * i) = constraint_linear.B;
            h_const.block<C, 1>(C * i, 0) = constraint_linear.c;

            lb_constraint_aug.template block<C, 1>(C * i, 0) = lb_constraint;
            ub_constraint_aug.template block<C, 1>(C * i, 0) = ub_constraint;

            lb_input_aug.template block<M, 1>(M * i, 0) = lb_input;
            ub_input_aug.template block<M, 1>(M * i, 0) = ub_input;
        }

        auto qf = quadratize_final<N, CostFinal>(cost_final, x0);
        Q_aug.template block<N, N>(N * (L - 2), N * (L - 2)) = qf.Q;
        q_aug.template block<1, N>(0, N * (L - 2)) = qf.q;

        Matrix<> cost_quadratic = R_aug + B_aug.transpose() * Q_aug * B_aug;
        Matrix<> cost_linear = r_aug + q_aug * B_aug + x_aug.transpose() * Q_aug * B_aug;

        Matrix<> H_aug = H_u + H_x * B_aug;
        lb_constraint_aug -= h_const - H_x * x_aug;
        ub_constraint_aug -= h_const - H_x * x_aug;

        qpOASES::Options options;
        options.printLevel = qpOASES::PL_NONE;
        problem.setOptions(options);

        int nWSR = 1000000;
        problem.init(cost_quadratic.data(), cost_linear.data(), H_aug.data(), lb_input_aug.data(), ub_input_aug.data(), lb_constraint_aug.data(), ub_constraint_aug.data(), nWSR, 0);
    }

    void iterate(Vector<N> x0) {
        Vector<> u0 = Vector<>::Zero(M * (L - 1));
        problem.getPrimalSolution(u0.data());

        // Do a forward pass similar to the above
        Matrix<> B_aug = Matrix<>::Zero(N * (L - 1), M * (L - 1));
        Matrix<> x_aug = Vector<>::Zero(N * (L - 1));

        Matrix<> Q_aug = Matrix<>::Zero(N * (L - 1), N * (L - 1));
        Matrix<> R_aug = Matrix<>::Zero(M * (L - 1), M * (L - 1));
        Matrix<> q_aug = Matrix<>::Zero(1, N * (L - 1));
        Matrix<> r_aug = Matrix<>::Zero(1, N * (L - 1));

        Matrix<> lb_input_aug = Vector<>::Zero(M * (L - 1));
        Matrix<> ub_input_aug = Vector<>::Zero(M * (L - 1));
        Matrix<> lb_constraint_aug = Vector<>::Zero(C * (L - 1));
        Matrix<> ub_constraint_aug = Vector<>::Zero(C * (L - 1));

        Matrix<N> A_i = Matrix<N>::Identity();

        Matrix<> B_aug_row = Matrix<>::Zero(N, M * (L - 1));

        Matrix<> H_x = Matrix<>::Zero(C * (L - 1), N * (L - 1));
        Matrix<> H_u = Matrix<>::Zero(C * (L - 1), M * (L - 1));
        Matrix<> h_const = Matrix<>::Zero(C * (L - 1), 1);

        Vector<N> unforced_x = x0;

        Vector<N> x_hat = x0;
        for (int i = 0; i < (L - 1); i++) {
            Vector<M> u_hat = u0.template block<M, 1>(M * i, 0);

            auto linearized = linearize<N, M, System>(system, x_hat, u_hat, i);
            B_aug_row = linearized.A * B_aug_row;
            B_aug_row.template block<N, M>(0, M * i) = linearized.B;
            B_aug.template block(N * i, 0, N, M * (L - 1)) = B_aug_row;
            unforced_x = linearized.A * unforced_x + linearized.c;
            x_aug.template block<N, 1>(N * i, 0) = unforced_x;

            x_hat = system(x_hat, u_hat, i);

            auto quadratized = quadratize<N, M, Cost>(cost, x_hat, u_hat, i);

            if (i != L - 2) {
                Q_aug.template block<N, N>(N * i, N * i) = quadratized.Q;
                q_aug.template block<1, N>(0, N * i) = quadratized.q;
            }

            R_aug.template block<M, M>(M * i, M * i) = quadratized.R;
            r_aug.template block<1, M>(0, M * i) = quadratized.r;

            auto constraint_linear = linearize<N, M, Constraint, C>(constraint, x_hat, u_hat, i);
            H_x.template block<C, N>(C * i, N * i) = constraint_linear.A;
            H_u.template block<C, M>(C * i, M * i) = constraint_linear.B;
            h_const.block<C, 1>(C * i, 0) = constraint_linear.c;

            lb_constraint_aug.template block<C, 1>(C * i, 0) = lb_constraint;
            ub_constraint_aug.template block<C, 1>(C * i, 0) = ub_constraint;

            lb_input_aug.template block<M, 1>(M * i, 0) = lb_input;
            ub_input_aug.template block<M, 1>(M * i, 0) = ub_input;
        }

        auto qf = quadratize_final<N, CostFinal>(cost_final, x0);
        Q_aug.template block<N, N>(N * (L - 2), N * (L - 2)) = qf.Q;
        q_aug.template block<1, N>(0, N * (L - 2)) = qf.q;

        Matrix<> cost_quadratic = R_aug + B_aug.transpose() * Q_aug * B_aug;
        Matrix<> cost_linear = r_aug + q_aug * B_aug + x_aug.transpose() * Q_aug * B_aug;

        Matrix<> H_aug = H_u + H_x * B_aug;
        lb_constraint_aug -= h_const - H_x * x_aug;
        ub_constraint_aug -= h_const - H_x * x_aug;

        int nWSR = 1000000;
        problem.hotstart(cost_quadratic.data(), cost_linear.data(), H_aug.data(), lb_input_aug.data(), ub_input_aug.data(), lb_constraint_aug.data(), ub_constraint_aug.data(), nWSR, 0);
    }

    Vector<> plan(Vector<N> x0, Vector<N> r) {
        for (int i = 0; i < 50; i++) {
             iterate(x0);
        }
        Vector<> u0 = Vector<>::Zero(M * (L - 1));
        problem.getPrimalSolution(u0.data());
        return u0;
    }

    Vector<M> control(Vector<N> x0, Vector<N> r) {
        return plan(x0, r).template block<M, 1>(0, 0);
    }

protected:
    qpOASES::SQProblem problem;

    System system;
    Constraint constraint;
    Cost cost;
    CostFinal cost_final;
    Vector<C> lb_constraint, ub_constraint;
    Vector<M> lb_input, ub_input;

    int L;
};
