#include "linear_elasticity.h"

double elastic_energy(const Eigen::VectorXd &u, const Eigen::SparseMatrix<double> &K) {
    return 0.5 * u.transpose() * K * u;
}

void elastic_gradient(const Eigen::VectorXd &u, const Eigen::SparseMatrix<double> &K, Eigen::VectorXd &g) {
    g = K * u;
}
void elastic_hessian(const Eigen::VectorXd &u, const Eigen::SparseMatrix<double> &K, Eigen::SparseMatrix<double> &H)
{
    H = K;
}
