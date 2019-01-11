#include "system.h"
#include "gtest/gtest.h"
#include <unsupported/Eigen/MatrixFunctions>

TEST(Discretized, SimpleLinearDiscretize) {
    auto continuous = [](Vector<2> x, Vector<1> u, double dt) {
        return Vector<2>(x(1), u(0) - x(0));
    };
    double dt = 0.01;
    auto discretized = discretize<2, 1>(continuous, dt);

    Vector<2> x0(1.0, 1.0);
    double T = 1.0;
    Matrix<2> Ad = ((Matrix<2>() << 0, 1, -1, 0).finished() * T).exp();
    Vector<2> x1 = Ad * x0;

    Vector<2> x = x0;
    for (double t = 0; t < T; t += dt) {
        x = discretized(x, Vector<1>(0), t);
    }

    EXPECT_NEAR(x(0), x1(0), 1e-5);
    EXPECT_NEAR(x(1), x1(1), 1e-5);
}

TEST(Discretized, NonlinearDiscretize) {
    auto continuous = [](Vector<2> x, Vector<1> u, double dt) {
        return Vector<2>(x(1), u(0) - std::sin(x(0)));
    };
    double dt = 0.01;
    auto discretized = discretize<2, 1>(continuous, dt);

    Vector<2> x0(-3.0, 0.0);

    double T = 10.0;

    Vector<2> x = x0;
    double xmax0 = x0(0);
    for (double t = 0; t < T; t += dt) {
        x = discretized(x, Vector<1>(0), t);

        xmax0 = std::max(x(0), xmax0);
    }

    EXPECT_NEAR(xmax0, 3.0, 1e-5);
}

TEST(Discretized, SimpleLinearDiscretizeReverse) {
    auto continuous = [](Vector<2> x, Vector<1> u, double dt) {
        return Vector<2>(x(1), u(0) - x(0));
    };
    double dt = -0.01;
    auto discretized = discretize<2, 1>(continuous, dt);

    Vector<2> x0(1.0, 1.0);
    double T = -1.0;
    Matrix<2> Ad = ((Matrix<2>() << 0, 1, -1, 0).finished() * T).exp();
    Vector<2> x1 = Ad * x0;

    Vector<2> x = x0;
    for (double t = 0; t > T; t += dt) {
        x = discretized(x, Vector<1>(0), t);
    }

    EXPECT_NEAR(x(0), x1(0), 1e-5);
    EXPECT_NEAR(x(1), x1(1), 1e-5);
}

TEST(Linearized, SimpleLinearization) {
    auto nonlinear = [](Vector<2> x, Vector<1> u, double dt) {
        return Vector<2>(x(1), u(0) - std::sin(x(0)));
    };
    Vector<2> x(0, 0);
    Vector<1> u(0);
    auto linear = linearize<2, 1>(nonlinear, x, u, 0.0);
    EXPECT_NEAR((linear.A - (Matrix<2>() << 0, 1, -1, 0).finished()).lpNorm<Eigen::Infinity>(), 0, 1e-5);
    EXPECT_NEAR((linear.B - (Matrix<2, 1>() << 0, 1).finished()).lpNorm<Eigen::Infinity>(), 0, 1e-5);
    EXPECT_NEAR(linear.c.lpNorm<Eigen::Infinity>(), 0, 1e-5);

    x(0) = 1;
    linear = linearize<2, 1>(nonlinear, x, u, 0.0);
    EXPECT_NEAR((linear.A - (Matrix<2>() << 0, 1, -std::cos(x(0)), 0).finished()).lpNorm<Eigen::Infinity>(), 0, 1e-5);
    EXPECT_NEAR((linear.B - (Matrix<2, 1>() << 0, 1).finished()).lpNorm<Eigen::Infinity>(), 0, 1e-5);
    EXPECT_NEAR((linear.c - (Vector<2>() << 0, x(0) * std::cos(x(0)) - std::sin(x(0))).finished()).lpNorm<Eigen::Infinity>(), 0, 1e-5);
}

TEST(Linearized, GeneralLinearization) {
    auto nonlinear = [](Vector<2> x, Vector<1> u, double dt) {
        return Vector<2>(std::cos(x(1)), std::cos(u(0)) - std::tanh(x(1) - x(0)));
    };

    constexpr int kNumSamples = 6;
    constexpr double kSampleOffset = kNumSamples / 2;
    for (int i = 0; i <= kNumSamples; i++) {
        for (int j = 0; j <= kNumSamples; j++) {
            for (int k = 0; k <= kNumSamples; k++) {
                Vector<2> x0((-kSampleOffset + i) / kSampleOffset, (-kSampleOffset + j) / kSampleOffset);
                Vector<1> u0((-kSampleOffset + k) / kSampleOffset);
                auto linearized = linearize<2, 1>(nonlinear, x0, u0, 0.0);
                for (int l = 0; l <= 4; l++) {
                    for (int m = 0; m <= 4; m++) {
                        for (int n = 0; n <= 4; n++) {
                            Vector<2> dx((-2 + l) / 10.0, (-2 + m) / 10.0);
                            Vector<1> du((-2 + n) / 10.0);
                            Vector<2> x = x0 + dx;
                            Vector<1> u = u0 + du;
                            Vector<2> xhat = linearized.A * x + linearized.B * u + linearized.c;
                            EXPECT_LT((xhat - nonlinear(x, u, 0)).lpNorm<Eigen::Infinity>() / (dx.norm() + du.norm() + 1e-5), 0.25);
                        }
                    }
                }
            }
        }
    }
}

TEST(Quadratized, SimpleQuadratization) {
    auto nonquadratic = [](Vector<2> x, Vector<1> u, double dt) {
        return std::exp(x(0) + x(0) * x(1) + x(1) * u(0));
    };
    Vector<2> x(0, 0);
    Vector<1> u(0);
    auto quadratized = quadratize(nonquadratic, x, u, 0);
    EXPECT_NEAR(quadratized.Q(0, 0), 1, 1e-3);
    EXPECT_NEAR(quadratized.Q(1, 0), 1, 1e-3);
    EXPECT_NEAR(quadratized.Q(0, 1), 1, 1e-3);
    EXPECT_NEAR(quadratized.q(0, 0), 1, 1e-3);
    EXPECT_NEAR(quadratized.P(1, 0), 1, 1e-3);
    EXPECT_NEAR(quadratized.q(1, 0), 0, 1e-3);
}

TEST(Quadratized, GeneralQuadratization) {
    auto nonquadratic = [](Vector<2> x, Vector<1> u, double dt) {
        return std::exp(x(0) * x(0)) + u(0) * u(0);
    };

    Vector<2> x(1, 1);
    Vector<1> u(1);
    auto quadratic = quadratize<2, 1>(nonquadratic, x, u, 0);

    constexpr int kNumSamples = 6;
    constexpr double kSampleOffset = kNumSamples / 2;
    for (int i = 0; i <= kNumSamples; i++) {
        for (int j = 0; j <= kNumSamples; j++) {
            for (int k = 0; k <= kNumSamples; k++) {
                Vector<2> x0((-kSampleOffset + i) / kSampleOffset, (-kSampleOffset + j) / kSampleOffset);
                Vector<1> u0((-kSampleOffset + k) / kSampleOffset);
                auto quadratized = quadratize<2, 1>(nonquadratic, x0, u0, 0.0);
                double c0 = nonquadratic(x0, u0, 0.0);
                for (int l = 0; l <= 2; l++) {
                    for (int m = 0; m <= 2; m++) {
                        for (int n = 0; n <= 4; n++) {
                            Vector<2> dx((-2 + l) / 10.0, (-2 + m) / 10.0);
                            Vector<1> du((-2 + n) / 10.0);
                            Vector<2> x = x0 + dx;
                            Vector<1> u = u0 + du;
                            double chat = x.dot(0.5 * quadratized.Q * x + quadratized.P * u + quadratized.q) + u.dot(0.5 * quadratized.R * u + quadratized.r) + quadratized.c;
                            EXPECT_NEAR((chat - nonquadratic(x, u, 0)) / (1e-5 + std::abs(nonquadratic(x, u, 0) - c0)), 0,  15 * (1e-3 + x.squaredNorm() + u.squaredNorm()));
                        }
                    }
                }
            }
        }
    }
}
