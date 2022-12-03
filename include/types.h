#ifndef COMP_DYN_REWRITE_TYPES_H
#define COMP_DYN_REWRITE_TYPES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

// modified file from CSC417 assignment setup
namespace Eigen {
    //dense types
    using Vector9d = Eigen::Matrix<double, 9, 1>;
    using Vector12d = Eigen::Matrix<double, 12,1>;

    using Matrix34d = Eigen::Matrix<double, 3,4>;
    using Matrix43d = Eigen::Matrix<double, 4,3>;
    using Matrix912d = Eigen::Matrix<double, 9, 12>;
    using Matrix99d = Eigen::Matrix<double, 9, 9>;
    using Matrix1212d = Eigen::Matrix<double, 12,12>;

    //sparse types
    using SparseMatrixd = Eigen::SparseMatrix<double>;
}


#endif //COMP_DYN_REWRITE_TYPES_H
