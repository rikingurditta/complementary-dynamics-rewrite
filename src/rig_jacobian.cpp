#include "rig_jacobian.h"

void rig_jacobian(const Eigen::MatrixXd &V, const Eigen::MatrixXd &W, Eigen::MatrixXd &J) {
    // maybe J should be sparse instead? it's gonna have a lot of 0s
    J.resize(V.rows() * 3, W.cols() * 12);
    for (int i = 0; i < V.rows(); i++) {
        for (int j = 0; j < W.cols(); j++) {
            // v1 = [v^T 1]
            Eigen::MatrixXd v1(1, 4);
            v1.block<1, 3>(0, 0) = V.row(i);
            v1(3) = 1;
            // calculate kronecker product of identity (*) v1
            Eigen::MatrixXd kronecker = Eigen::MatrixXd::Zero(3, 12);
            kronecker.block<1, 4>(0, 0) = v1;
            kronecker.block<1, 4>(1, 4) = v1;
            kronecker.block<1, 4>(2, 8) = v1;
            // each block of J is w_ij * Id (*) [v^T 1]
            J.block<3, 12>(i * 3, j * 12) = W(i, j) * kronecker;
        }
    }
}