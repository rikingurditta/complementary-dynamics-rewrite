#ifndef COMP_DYN_REWRITE_LINEAR_ELASTICITY_H
#define COMP_DYN_REWRITE_LINEAR_ELASTICITY_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

double elastic_energy(const Eigen::VectorXd &u, const Eigen::SparseMatrix<double> &K);
void elastic_gradient(const Eigen::VectorXd &u, const Eigen::SparseMatrix<double> &K, Eigen::VectorXd &g);
void elastic_hessian(const Eigen::VectorXd &u, const Eigen::SparseMatrix<double> &K, Eigen::SparseMatrix<double> &H);

#endif //COMP_DYN_REWRITE_LINEAR_ELASTICITY_H
