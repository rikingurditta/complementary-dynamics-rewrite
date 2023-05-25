#include <igl/opengl/glfw/Viewer.h>
#include "types.h"
#include "igl/lbs_matrix.h"
#include "igl/bounding_box.h"
#include "complementary_displacement.h"
#include "igl/parula.h"
#include "lbs_rig_jacobian.h"

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

    // constants
    double dt = 0.1;
    double k = 100; // stiffness
    double g = -9.8; // acceleration due to gravity

    // load tet mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi T;
    Eigen::MatrixXi F;
    igl::readMESH("../examples/fish/fish.mesh", V, T, F);
    long n = V.rows();
    igl::boundary_facets(T, F);
    // boundary_facets returns inside out surface, so reverse direction of every face to turn the correct side outward
    Eigen::VectorXi temp = F.col(0);
    F.col(0) = F.col(1);
    F.col(1) = temp;

    // mass matrix
    Eigen::SparseMatrixd M(n * 3, n * 3);
    M.setIdentity();

    // linear blend skinning
    Eigen::MatrixXd bones(2, 3);
    bones << 0, 2, -3,
            0, 0, 1;
    Eigen::MatrixXd weights(n, 2);
    for (int i = 0; i < n; i++) {
        weights(i, 0) = 100 / (V.row(i) - bones.row(0)).squaredNorm();
        weights(i, 1) = 100 / (V.row(i) - bones.row(1)).squaredNorm();
        double s = weights.row(i).sum();
        weights.row(i) /= s;
    }
    Eigen::MatrixXd LBS;
    igl::lbs_matrix(V, weights, LBS);
    std::cout << "LBS: " <<  LBS.rows() << ", " << LBS.cols() << "\n";

    // calculate rig jacobian
    Eigen::MatrixXd J;
    lbs_rig_jacobian(V, weights, J);

    int num_frames = 48 * 2;

    // create bone transformations matrices for each frame of animation
    std::vector<Eigen::MatrixXd> T_list;
    for (int i = 0; i < num_frames; i++) {
        Eigen::Matrix34d T1, T2;
        double t = (double) i / num_frames;
        double angle = abs(M_PI / 3 * (0.5 - t));
        T1 << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, t;
        T2 << cos(angle), -sin(angle), 0, 0,
                sin(angle), cos(angle), 0, 0,
                0, 0, 1, t;
        Eigen::Matrix<double, 8, 3> T_curr;
        T_curr.block<4, 3>(0, 0) = T1.transpose();
        T_curr.block<4, 3>(4, 0) = T2.transpose();
        T_list.emplace_back(T_curr);
    }

    Eigen::VectorXd u_prev_prev = Eigen::VectorXd::Zero(n * 3);
    Eigen::VectorXd u_prev = Eigen::VectorXd::Zero(n * 3);
    Eigen::VectorXd du_prev = Eigen::VectorXd::Zero(n * 3);

    Eigen::SimplicialLDLT<Eigen::SparseMatrixd> solver;
    cd_precompute(V, T, M, k, J, dt, solver);

    // view animation
    bool only_visualize_complementary = false;
    int frame = 0;

    igl::opengl::glfw::Viewer viewer;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool {
        if (viewer.core().is_animating) {
            // current transformation
            Eigen::MatrixXd T_curr = T_list[frame];
            // get rig deformation using linear blend skinning
            Eigen::MatrixXd ur = flatten(LBS * T_curr - V);

            // apply external impulse to certain vertices at beginning of animation
            Eigen::VectorXd ft = Eigen::VectorXd::Zero(n * 3);
            std::list<int> vertices{300};
            double f0 = 2000;
            if (frame == 0) {
                for (int v: vertices) {
                    ft(v * 3 + 0) = f0;
                }
            }

            // get complementary displacement
            Eigen::VectorXd uc = Eigen::VectorXd::Zero(ur.size());
            complementary_displacement(V, T, M, k, ur, u_prev, du_prev, J, ft, dt, solver, uc);
            u_prev_prev = u_prev;
            u_prev = ur + uc;
            du_prev = (u_prev - u_prev_prev) / dt;
            std::cout << "||uc||²: " << uc.squaredNorm() << "\n";
            std::cout << "||ur + uc||²: " << (ur + uc).squaredNorm() << "\n";
            if (only_visualize_complementary)
                viewer.data(0).set_vertices(V + unflatten(uc, 3));
            else
                viewer.data(0).set_vertices(V + unflatten(ur + uc, 3));

            frame++;
            if (frame == num_frames) {
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
                only_visualize_complementary = !only_visualize_complementary;
                std::cout << "only_visualize_complementary " << only_visualize_complementary << "\n";
                break;
            default:
                break;
        }
        return false;
    };

    viewer.core().is_animating = false;
    viewer.core().animation_max_fps = 16.;
    viewer.data(0).show_lines = false;
//    viewer.data(0).show_faces = false;
    viewer.data(0).show_overlay_depth = false;
    viewer.data(0).set_mesh(V, F);
    viewer.append_mesh();
    Eigen::MatrixXd p(2, 3);
    p << 3, 2, -3,
            3, 0, 1;
    Eigen::MatrixXd c = Eigen::MatrixXd::Ones(2, 3);
    viewer.data(1).set_points(p, c);

    viewer.launch();

    return 0;
}
