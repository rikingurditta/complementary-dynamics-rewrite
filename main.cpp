#include <igl/opengl/glfw/Viewer.h>
#include "types.h"
#include "igl/lbs_matrix.h"
#include "rig_jacobian.h"
#include "igl/bounding_box.h"
#include "complementary_displacement.h"
#include "igl/parula.h"

Eigen::VectorXd flatten(const Eigen::MatrixXd &V) {
    long rows = V.rows(), cols = V.cols();
    Eigen::VectorXd out(rows * cols);
    for (long r = 0; r < rows; r++) {
        out.segment(r * cols, cols) = V.row(r);
    }
    return out;
}

Eigen::MatrixXd unflatten(const Eigen::VectorXd &v, int dim) {
    long rows = v.rows() / dim, cols = dim;
    Eigen::MatrixXd out(rows, cols);
    for (long r = 0; r < rows; r++) {
        out.row(r) = v.segment(r * cols, cols);
    }
    return out;
}

int main(int argc, char *argv[]) {
    // load tet mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi T;
    Eigen::MatrixXi F;
    igl::readMESH("../examples/fish/fish.mesh", V, T, F);
    igl::boundary_facets(T, F);
    // boundary_facets returns inside out surface, so reverse direction of every face to turn the correct side outward
    Eigen::VectorXi temp = F.col(0);
    F.col(0) = F.col(1);
    F.col(1) = temp;

    long n = V.rows();

    // choose two bones based on corners of bounding box
    // ideally bones would be chosen by animator in a 3d modelling program, but the transfer of data was too difficult,
    // so here we manually place bones
    int NUM_BONES = 2;
    Eigen::MatrixXd BV;
    Eigen::MatrixXd BF;
    igl::bounding_box(V, BV, BF);
    Eigen::RowVector3d lo = (BV.row(2) + BV.row(3)) / 2;
    Eigen::RowVector3d ro = (BV.row(4) + BV.row(5)) / 2;
    Eigen::RowVector3d l = (3 * lo + ro) / 4;
    Eigen::RowVector3d r = (3 * ro + lo) / 4;
    Eigen::MatrixXd bones(NUM_BONES, 3);
    bones.row(0) = l;
    bones.row(1) = r;

    // create bone transformations matrices for each frame of animation
    // again, ideally this would be created by animator in modelling program, but the data transfer was too difficult
    std::vector<Eigen::MatrixXd> T_list;
    for (int i = 0; i < 48; i++) {
        Eigen::Matrix34d T1 = Eigen::Matrix34d::Zero();
        T1.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
//        Eigen::Vector3d d = Eigen::Vector3d::Zero();
//        d << i, 0, 0;  // displacement increases with frame, creating translation animation
//        T1.block<3, 1>(0, 3) = d * 2;
        Eigen::Matrix34d T2 = Eigen::Matrix34d::Zero();
        double angle = (i - 24) * 2. * M_PI / 180;  // angle increases with frame, creating rotation animation
        T2.block<3, 3>(0, 0) << 1, 0, 0,
                0, cos(angle), -sin(angle),
                0, sin(angle), cos(angle);
        // T2.block<3, 1>(0, 3) = d * 2;
        Eigen::Matrix<double, 8, 3> T_curr;
        T_curr.block<4, 3>(0, 0) = T1.transpose();
        T_curr.block<4, 3>(4, 0) = T2.transpose();
        T_list.emplace_back(T_curr);
    }

    // temporary: calculate weights based on inverse squared distance
    // in actual setup, weights are chosen by animator
    // we haven't figured out how to export weights from Blender
    Eigen::MatrixXd weights(V.rows(), 2);
    for (int i = 0; i < V.rows(); i++) {
        weights(i, 0) = 100 / (V.row(i) - l).squaredNorm();
        weights(i, 1) = 100 / (V.row(i) - r).squaredNorm();
        double s = weights.row(i).sum();
        weights.row(i) /= s;
    }

    // calculate skinning matrix
    Eigen::MatrixXd LBS;
    igl::lbs_matrix(V, weights, LBS);

    // calculate rig jacobian
    Eigen::MatrixXd J;
    rig_jacobian(V, weights, J);

    // blend shapes
    Eigen::MatrixXd B(n * 3, 3);
    Eigen::VectorXd b0 = flatten(V);
    B.col(0) = b0;
    B.col(1) = b0 + Eigen::VectorXd::Ones(n * 3);
    Eigen::VectorXd disp2 = Eigen::VectorXd::Ones(n * 3);
    for (int i = 0; i < n; i++) {
        disp2(i * 3 + 1) = -1;
        disp2(i * 3 + 2) = 0;
    }
    B.col(2) = b0 - disp2 * 3;

    double dt = 10000;

    //material parameters
    double density = 0.1;
    double YM = 6e5; //young's modulus
    double mu = 0.4; //poissons ratio
    double neohookean_D = 0.5 * (YM * mu) / ((1.0 + mu) * (1.0 - 2.0 * mu));
    double neohookean_C = 0.5 * YM / (2.0 * (1.0 + mu));
    std::cout << "C: " << neohookean_C << " D: " << neohookean_D << "\n";

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;

    Eigen::SparseMatrixd M(n * 3, n * 3);
//    igl::massmatrix(V, T, igl::MassMatrixType::MASSMATRIX_TYPE_DEFAULT, M);
    std::cout << "n " << n << " n * 3 " << n * 3 << "\n";
    std::cout << "M dim " << M.rows() << ", " << M.cols() << "\n";
    M.setIdentity();

    Eigen::VectorXd u_prev_prev = Eigen::VectorXd::Zero(n * 3);
    Eigen::VectorXd u_prev = Eigen::VectorXd::Zero(n * 3);
    Eigen::VectorXd du_prev = Eigen::VectorXd::Zero(n * 3);

    bool oc = false;
    int frame = 0;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool {
        if (viewer.core().is_animating) {
            // current transformation
            Eigen::MatrixXd T_curr = T_list[frame];
            // get rig deformation using linear blend skinning
//            Eigen::MatrixXd Ur = LBS * T_curr - V;
//            Eigen::VectorXd ur = flatten(Ur);

            // get rig deformation using blend shapes
            Eigen::Vector3d mix;
            double t = frame / 24.;
            if (t < 1)
                mix << 1 - t, t, 0;
            else
                mix << 0, 1 - (t - 1), t - 1;
            Eigen::VectorXd ur = B * mix - flatten(V);

            // get complementary displacement
            Eigen::VectorXd uc;
            complementary_displacement(V, T, M, ur, u_prev, du_prev, B, dt, uc);
            u_prev_prev = u_prev;
            u_prev = ur + uc;
            du_prev = (u_prev - u_prev_prev) / dt;
            std::cout << "||uc||²: " << uc.squaredNorm() << "\n";
            if (oc) {
                // only visualize complementary displacement
                viewer.data().set_vertices(V + unflatten(uc, 3));
            } else {
                viewer.data().set_vertices(V + unflatten(ur + uc, 3));
            }

            frame++;
            if (frame == T_list.size()) {
                frame = 0;
                viewer.core().is_animating = false;
            }
        }
        return false;
    };
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod) {
        std::cout << key << "\n";
        switch (key) {
            case ' ':
                std::cout << "space" << "\n";
                viewer.core().is_animating = !viewer.core().is_animating;
                break;
            case 'c':
            case 'C':
                oc = !oc;
                std::cout << "oc " << oc << "\n";
                break;
            default:
                break;
        }
        return false;
    };

    viewer.data().set_mesh(V, F);
    viewer.core().is_animating = false;
    viewer.core().animation_max_fps = 16.;
    viewer.data().show_lines = false;
    viewer.data().show_overlay_depth = false;
    viewer.data().set_mesh(V, F);

    // colour based on bones - will need to change this when we have more than 2 bones
    Eigen::MatrixXd C;
    igl::parula(weights.col(0), true, C);
    viewer.data().set_colors(C);

    // visualize handles as points
    const Eigen::RowVector3d orange(1.0, 0.7, 0.2);
    viewer.data().set_points(bones, orange);

    viewer.launch();

    return 0;
}
