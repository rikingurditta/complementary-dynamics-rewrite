#include <igl/opengl/glfw/Viewer.h>
#include "types.h"
#include "igl/lbs_matrix.h"
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

    double k = 100; // stiffness
    double g = -9.8; // acceleration due to gravity

    long n = V.rows();

    // blend shapes
    // columns of J are blend shapes, J itself is the rig jacobian
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n * 3, 4);
    Eigen::VectorXd b0 = flatten(V);
    J.col(0) = b0;

    Eigen::VectorXd disp = Eigen::VectorXd::Zero(n * 3);
    int query_axis = 2;
    int disp_axis = 1;
    double min_coord = V(0, query_axis);
    for (int i = 0; i < n; i++) {
        if (V(i, query_axis) < min_coord)
            min_coord = V(i, query_axis);
    }
    // points that are further on query_axis get displaced more
    for (int i = 0; i < n; i++) {
        disp(i * 3 + disp_axis) = V(i, query_axis) - min_coord;
    }
    J.col(1) = b0 + disp * 2;
    J.col(2) = b0;
    J.col(3) = b0 + disp;

    double dt = 0.1;

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;

    Eigen::SparseMatrixd M(n * 3, n * 3);
    M.setIdentity();

    Eigen::VectorXd u_prev_prev = Eigen::VectorXd::Zero(n * 3);
    Eigen::VectorXd u_prev = Eigen::VectorXd::Zero(n * 3);
    Eigen::VectorXd du_prev = Eigen::VectorXd::Zero(n * 3);

    bool only_visualize_complementary = false;
    int frame = 0;
    int num_frames = 48;

    Eigen::SimplicialLDLT<Eigen::SparseMatrixd> solver;
    cd_precompute(V, T, M, k, J, dt, solver);
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool {
        if (viewer.core().is_animating) {
            // get rig deformation using blend shapes
            Eigen::Vector4d mix;
            double t = frame * 3. / num_frames;
            std::cout << "t " << t << "\n";
            if (t < 1)
                mix << 1 - t, t, 0, 0;
            else if (t < 2)
                mix << 0, 1 - (t - 1), t - 1, 0;
            else
                mix << 0, 0, 1 - (t - 2), t - 2;
            Eigen::VectorXd ur = J * mix - flatten(V);

            Eigen::VectorXd ft = Eigen::VectorXd::Zero(n * 3);
            std::list<int> vertices {300};
            double f0 = 10000;
            if (t == 0) {
                for (int v : vertices) {
//                    ft(v * 3 + 0) = f0;
//                    ft(v * 3 + 1) = f0;
//                    ft(v * 3 + 2) = f0;
                    ft(v * 3 + disp_axis) = f0;
                }
            }
//            for (int v = 0; v < n; v++) {
//                ft(v * 3 + disp_axis) = g;
//            }

            // get complementary displacement
            Eigen::VectorXd uc = Eigen::VectorXd::Zero(ur.size());
            complementary_displacement(V, T, M, k, ur, u_prev, du_prev, J, ft, dt, solver, uc);
            u_prev_prev = u_prev;
            u_prev = ur + uc;
            du_prev = (u_prev - u_prev_prev) / dt;
            std::cout << "||uc||²: " << uc.squaredNorm() << "\n";
            if (only_visualize_complementary)
                viewer.data().set_vertices(V + unflatten(uc, 3));
            else
                viewer.data().set_vertices(V + unflatten(ur + uc, 3));

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

    viewer.data().set_mesh(V, F);
    viewer.core().is_animating = false;
    viewer.core().animation_max_fps = 16.;
    viewer.data().show_lines = false;
    viewer.data().show_overlay_depth = false;
    viewer.data().set_mesh(V, F);

    // colour based on bones - will need to change this when we have more than 2 bones
//    Eigen::MatrixXd C;
//    igl::parula(weights.col(0), true, C);
//    viewer.data().set_colors(C);

    viewer.launch();

    return 0;
}
