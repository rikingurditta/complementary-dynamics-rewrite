#ifndef COMP_DYN_REWRITE_LBS_RIG_JACOBIAN_H
#define COMP_DYN_REWRITE_LBS_RIG_JACOBIAN_H

#include <Eigen/Core>

void lbs_rig_jacobian(const Eigen::MatrixXd &V, const Eigen::MatrixXd &weights, Eigen::MatrixXd &J);

#endif //COMP_DYN_REWRITE_LBS_RIG_JACOBIAN_H
