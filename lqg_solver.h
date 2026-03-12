#pragma once
// ============================================================
// Decentralized LQG noise-state game solver
// F-free optimized solver (no 36MB Kernel3D in filter/control)
// ============================================================

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------- compile-time constants ----------
constexpr int T_VAL = 1;
constexpr int N = 40;
constexpr double DT = static_cast<double>(T_VAL) / (N - 1);
constexpr double RHO = 0.1;
constexpr double X0 = 0.0;
constexpr double TERMINAL_STATE_WEIGHT = 0.0;
constexpr double B1_DEFAULT = 1.0;
constexpr double B2_DEFAULT = -1.0;
constexpr int D_W = 3;

constexpr int FORWARD_INNER_ITERS = 5;
constexpr int FILTER_INNER_ITERS = 8;
constexpr double FILTER_RELAX = 0.55;
constexpr int MAX_PICARD_ITERS = 10000;
constexpr double PICARD_RELAX = 0.15;
constexpr double PICARD_TOL = 1e-5;

// Mutable global targets (modified for cost experiments)
extern double g_b1;
extern double g_b2;

// ---------- type aliases ----------
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using VecN = Eigen::VectorXd;

// 2D kernel: shape (N, N, D_W) stored as array of arrays of Vec3
// ~N*N*3*8 = ~38KB — fine on stack
using Kernel2D = std::array<std::array<Vec3, N>, N>;

// 3D kernel: shape (N, N, N, D_W, D_W) → F[t][u][s] is a 3x3 matrix
// ~N*N*N*9*8 = ~36MB — must be heap-allocated
using Kernel3DRaw = std::array<std::array<std::array<Mat3, N>, N>, N>;

struct Kernel3D {
    std::unique_ptr<Kernel3DRaw> data;

    Kernel3D() : data(std::make_unique<Kernel3DRaw>()) {}
    Kernel3D(const Kernel3D& o) : data(std::make_unique<Kernel3DRaw>(*o.data)) {}
    Kernel3D(Kernel3D&&) noexcept = default;
    Kernel3D& operator=(const Kernel3D& o) {
        if (this != &o) data = std::make_unique<Kernel3DRaw>(*o.data);
        return *this;
    }
    Kernel3D& operator=(Kernel3D&&) noexcept = default;

    auto& operator[](int i) { return (*data)[i]; }
    const auto& operator[](int i) const { return (*data)[i]; }
};

// ---------- t_grid ----------
inline std::array<double, N> make_t_grid() {
    std::array<double, N> g;
    for (int i = 0; i < N; ++i)
        g[i] = static_cast<double>(i) * T_VAL / (N - 1);
    return g;
}
inline const std::array<double, N>& t_grid() {
    static auto g = make_t_grid();
    return g;
}

// ---------- fixed matrices ----------
inline Vec3 E0() {
    return Vec3(1.0, 0.0, 0.0);
}

inline Mat3 Pi1() {
    Mat3 m = Mat3::Zero();
    m(1, 1) = 1.0;
    return m;
}

inline Mat3 Pi2() {
    Mat3 m = Mat3::Zero();
    m(2, 2) = 1.0;
    return m;
}

// ---------- environment result ----------
struct EnvironmentResult {
    Kernel2D X;
    Kernel2D R1, R2;
    Kernel2D calD1, calD2;
    // Rank-1 filter factorization: A_store[k][s] for s < k
    Kernel2D A_store1, A_store2;
    // Observation parameters (needed for F materialization)
    double obs_gain1, obs_gain2;
    int obs_idx1, obs_idx2;
};

// ---------- bar solution ----------
struct BarSolution {
    std::array<double, N> barX;
    std::array<double, N> barD1;
    std::array<double, N> barD2;
    double bar_residual;
};

// ---------- backward adjoint result ----------
struct BackwardBarResult {
    std::array<double, N> barHx;
    Kernel2D barHk; // (N, N, D_W)
};

// ---------- functions ----------
void state_kernel_from_calD(const Kernel2D& calD1, const Kernel2D& calD2,
                            Kernel2D& X);

void forward_environment(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env);

// Hk-free version: uses internal ping-pong buffers (~230KB) instead of 36MB Kernel3D
void backward_kernels(const Kernel2D& X, const Kernel2D& Rk,
                      const Kernel2D& Dk, const std::array<double, N>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx);

// Legacy version that also fills Hk (only needed for figure output)
void backward_kernels(const Kernel2D& X, const Kernel2D& Rk,
                      const Kernel2D& Dk, const std::array<double, N>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel3D& Hk);

BackwardBarResult backward_bar_adjoints(
    const Kernel2D& X, const Kernel2D& Rk, const Kernel2D& Dk,
    const std::array<double, N>& barX, double b,
    const std::array<double, N>& prec_k, double terminal_weight);

BarSolution solve_bar_equilibrium(
    const EnvironmentResult& env, const Kernel2D& D1, const Kernel2D& D2,
    double prec1, double prec2,
    int max_iters = 2000, double relax = 0.08, double tol = 1e-10,
    bool verbose = false);

struct EquilibriumResult {
    Kernel2D D1, D2;
    EnvironmentResult env;
    std::vector<double> residuals;
};

EquilibriumResult solve_equilibrium(
    double p1_val, double p2_val, bool verbose = true,
    const Mat3& Pi_1 = Pi1(), int obs_idx_1 = 1,
    const Mat3& Pi_2 = Pi2(), int obs_idx_2 = 2);

// ---------- cost computation ----------
struct CostPair {
    double J1, J2;
};

CostPair compute_costs_general(const EnvironmentResult& env,
                               const BarSolution& bar_sol,
                               double r_val, double b1_val, double b2_val);

// ---------- F materialization (for figure output only) ----------
// Builds F[j][u][s] for ALL j from R + A_store. Writes into pre-allocated Kernel3D.
void materialize_F(const Kernel2D& R, const Kernel2D& A_store,
                   double obs_gain, int obs_index, Kernel3D& F);

// ---------- utility ----------
inline std::array<double, N> make_constant_prec(double val) {
    std::array<double, N> a;
    a.fill(val);
    return a;
}
