#include <iostream>
#include "complementary_displacement.h"

void cd_precompute(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, const Eigen::SparseMatrixd &M, double k,
                   const Eigen::MatrixXd &J, double dt, Eigen::SimplicialLDLT<Eigen::SparseMatrixd> &solver) {
    long n = V.rows();
    long m = J.cols();

    // use linear elasticity
    Eigen::SparseMatrixd K(n * 3, n * 3);
    K.setIdentity();
    K *= k;
    auto f = [=](const Eigen::VectorXd &u, const Eigen::SparseMatrixd &K) {
        return 0.5 * u.transpose() * K * u;
    };
    auto g = [=](const Eigen::VectorXd &u, const Eigen::SparseMatrixd &K, Eigen::VectorXd &grad) {
        grad = K * u;
    };
    auto H = [=](const Eigen::VectorXd &u, const Eigen::SparseMatrixd &K, Eigen::SparseMatrixd &Hess) {
        Hess = K;
    };

    Eigen::MatrixXd MTJ = M.transpose() * J;

    Eigen::SparseMatrixd A(n * 3 + m, n * 3 + m);
    // A = [K + M/dt^2    M^T J
    //      (M^T J)^T     0     ]
    std::vector<Eigen::Triplet<double>> tl;
    for (int i = 0; i < K.outerSize(); i++) {
        for (Eigen::SparseMatrixd::InnerIterator it(K, i); it; ++it) {
            tl.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int i = 0; i < M.outerSize(); i++) {
        for (Eigen::SparseMatrixd::InnerIterator it(M, i); it; ++it) {
            tl.emplace_back(it.row(), it.col(), it.value() / (dt * dt));
        }
    }
    for (int i = 0; i < MTJ.rows(); i++) {
        for (int j = 0; j < MTJ.cols(); j++) {
            tl.emplace_back(i, n * 3 + j, MTJ(i, j));
            tl.emplace_back(n * 3 + j, i, MTJ(i, j));
        }
    }
    A.setFromTriplets(tl.begin(), tl.end());

    solver.compute(A);
}


void
complementary_displacement(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, const Eigen::SparseMatrixd &M, double k,
                           const Eigen::VectorXd &ur, const Eigen::VectorXd &u_prev, const Eigen::VectorXd &du_prev,
                           const Eigen::MatrixXd &J, const Eigen::VectorXd &ft, double dt,
                           Eigen::SimplicialLDLT<Eigen::SparseMatrixd> &solver, Eigen::VectorXd &uc) {
    long n = V.rows();
    long m = J.cols();

    // use linear elasticity
    Eigen::SparseMatrixd K(n * 3, n * 3);
    K.setIdentity();
    K *= k;

    Eigen::VectorXd b = Eigen::VectorXd::Zero(n * 3 + m);
    b.head(n * 3) = -K * ur - M * ((ur - u_prev) / dt - du_prev) / dt + ft;

    Eigen::VectorXd sol = solver.solve(b);
    uc = sol.head(n * 3);
}