#include <lbs_rig_jacobian.h>

void lbs_rig_jacobian(const Eigen::MatrixXd &V, const Eigen::MatrixXd &weights, Eigen::MatrixXd &J) {
    J = Eigen::MatrixXd::Zero(V.rows() * 3, weights.cols() * 12);
    for (int i = 0; i < V.rows(); i++) {
        for (int j = 0; j < weights.cols(); j++) {
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
            J.block<3, 12>(i * 3, j * 12) = weights(i, j) * kronecker;
        }
    }
}