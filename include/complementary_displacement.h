#ifndef COMP_DYN_REWRITE_COMPLEMENTARY_DISPLACEMENT_H
#define COMP_DYN_REWRITE_COMPLEMENTARY_DISPLACEMENT_H

#include "types.h"

void complementary_displacement(const Eigen::MatrixXd &V,
                                const Eigen::MatrixXi &T,
                                const Eigen::SparseMatrixd &M,
                                const Eigen::VectorXd &ur,
                                const Eigen::VectorXd &u_prev,
                                const Eigen::VectorXd &du_prev,
                                const Eigen::MatrixXd &J,
                                double dt,
                                Eigen::VectorXd &uc);

#endif //COMP_DYN_REWRITE_COMPLEMENTARY_DISPLACEMENT_H
