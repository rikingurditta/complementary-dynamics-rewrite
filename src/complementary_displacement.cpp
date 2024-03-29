#include <iostream>
#include "complementary_displacement.h"
#include "igl/boundary_facets.h"

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

    Eigen::MatrixXi F;
    igl::boundary_facets(T, F);
    Eigen::VectorXd d = Eigen::VectorXd::Ones(n * 3);
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < F.cols(); j++) {
            int v = F(i, j);
            d.segment<3>(v * 3) << 0, 0, 0;
        }
    }

    Eigen::MatrixXd MTJ = d.asDiagonal() * M.transpose() * J;

    // see eqn (11) in paper
    Eigen::SparseMatrixd A(n * 3 + m, n * 3 + m);
    // A = [K + M/dt^2    M^T J
    //        J^T M         0  ]
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

    // see eqn (11) in paper
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n * 3 + m);
    b.head(n * 3) = -K * ur - M / dt * ((ur - u_prev) / dt - du_prev) + ft;

    Eigen::VectorXd sol = solver.solve(b);
    uc = sol.head(n * 3);
}