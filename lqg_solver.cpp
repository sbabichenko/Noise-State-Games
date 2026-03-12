// ============================================================
// Decentralized LQG noise-state game solver — F-free implementation
//
// Key optimization: the 36MB Kernel3D F is never materialized during
// the solver loop. Instead, products like sum_u F[j][u][s]^T * v[u]
// are computed directly from the rank-1 decomposition:
//   F[j][u][s] = border[u][s] + sum_k R[k][u] * A_store[k][s]^T
//
// This eliminates the dominant O(N^3) F copy per filter call and
// removes 72MB of heap allocation from EnvironmentResult.
// ============================================================

#include "lqg_solver.h"
#include <algorithm>

double g_b1 = B1_DEFAULT;
double g_b2 = B2_DEFAULT;

// ============================================================
// state_kernel_from_calD
// ============================================================
void state_kernel_from_calD(const Kernel2D& calD1, const Kernel2D& calD2,
                            Kernel2D& X) {
    for (auto& row : X)
        for (auto& v : row)
            v.setZero();

    for (int s = 0; s < N; ++s) {
        X[s][s] = E0();
        Vec3 cumsum = Vec3::Zero();
        for (int t = s + 1; t < N; ++t) {
            cumsum += DT * (calD1[t - 1][s] + calD2[t - 1][s]);
            X[t][s] = E0() + cumsum;
        }
    }
}

// ============================================================
// compute_filter_kernels — F-free implementation
//
// Computes R[j][s] and A_store[j][s] without materializing
// the 3D filter kernel F.  Xhat is not computed since X = Xhat + R
// makes it redundant (and it is unused downstream).
//
// Xhat_const is computed from A_store history:
//   Xhat_const[s] = Pi*X[j][s] + gamma_a[s]
//                 + DT*g*e_i * sum_{u<s} dot(R[s][u], X[j][u])
//                 + DT * sum_{k=s+1}^{j-1} partial_c_k * A_store[k][s]
//
// where gamma_a[s] = border_gt contribution (already computed for Gamma)
//       partial_c_k = c_k - dot(R[k][k], X[j][k])
// ============================================================
static void compute_filter_kernels(
    const Kernel2D& X, const Mat3& Pi, int obs_index,
    double obs_gain_val,
    int filter_iters, double relax,
    Kernel2D& R, Kernel2D& A_store) {

    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    Mat3 I_minus_Pi = Mat3::Identity() - Pi;
    double g = obs_gain_val;
    double prec = g * g;
    double dt2_prec = DT * DT * prec;

    for (int j = 0; j < N; ++j) {
        int m = j + 1;

        if (j == 0) {
            // Base case: no history
            R[0][0] = I_minus_Pi * X[0][0];
            for (int s = 1; s < N; ++s) R[0][s].setZero();
            A_store[0][0].setZero();
            continue;
        }

        // --- Compute c_k and partial_c_k for k = 0..j-1 ---
        // c_k = sum_{u=0}^{k} dot(R[k][u], X[j][u])
        // partial_c_k = c_k - dot(R[k][k], X[j][k]) = sum_{u=0}^{k-1} dot(R[k][u], X[j][u])
        std::array<double, N> c_k, partial_c_k;
        for (int k = 0; k < j; ++k) {
            double acc = 0.0;
            for (int u = 0; u < k; ++u)
                acc += R[k][u].dot(X[j][u]);
            partial_c_k[k] = acc;
            c_k[k] = acc + R[k][k].dot(X[j][k]);
        }

        // --- coeff_a for gamma_a ---
        // coeff_a[z] = X[j][z](obs_index) * g
        std::array<double, N> coeff_a;
        for (int z = 0; z < j; ++z)
            coeff_a[z] = X[j][z](obs_index) * g;

        // --- Merged computation per s: Q_corr + gamma_a + Xhat interior ---
        // All iterate k = s+1..j-1, accessing R[k][s] and A_store[k][s]
        std::array<Vec3, N> correction;
        std::array<Vec3, N> Xhat_const;

        for (int s = 0; s < j; ++s) {
            Vec3 q_upper = Vec3::Zero();
            Vec3 ga_acc = Vec3::Zero();
            Vec3 xi_acc = Vec3::Zero();

            for (int k = s + 1; k < j; ++k) {
                Vec3 Rks = R[k][s];
                q_upper += c_k[k] * Rks;
                ga_acc += coeff_a[k] * Rks;
                xi_acc += partial_c_k[k] * A_store[k][s];
            }

            Vec3 Q_corr_s = dt2_prec * (c_k[s] * R[s][s] + q_upper);
            Vec3 gamma_a_s = DT * ga_acc;

            // gamma_b iterates u < s; border_lt reuses partial_c_k[s]
            double gb_acc = 0.0;
            for (int u = 0; u < s; ++u)
                gb_acc += X[j][u].dot(R[u][s]);

            correction[s] = Q_corr_s + gamma_a_s + (DT * gb_acc * g) * e_i;
            Xhat_const[s] = Pi * X[j][s] + gamma_a_s
                          + (DT * partial_c_k[s] * g) * e_i
                          + DT * xi_acc;
        }

        // s = j: only gamma_b contributes to correction
        {
            double gb_acc = 0.0;
            for (int z = 0; z < j; ++z)
                gb_acc += X[j][z].dot(R[z][j]);
            correction[j] = (DT * gb_acc * g) * e_i;
        }

        // --- R[j][s] = (I - Pi) * X[j][s] - correction[s] ---
        for (int s = 0; s < m; ++s)
            R[j][s] = I_minus_Pi * X[j][s] - correction[s];
        for (int s = m; s < N; ++s)
            R[j][s].setZero();

        // --- Rank-1 filter iteration ---
        double sigma = 0.0;
        for (int u = 0; u < j; ++u)
            sigma += R[j][u].dot(DT * X[j][u]);

        double alpha = relax * DT * prec;
        double beta = 1.0 - relax;

        std::array<Vec3, N> A;
        for (int s = 0; s < j; ++s)
            A[s].setZero();

        for (int iter = 0; iter < filter_iters; ++iter) {
            for (int s = 0; s < j; ++s) {
                Vec3 diff = X[j][s] - Xhat_const[s] - sigma * A[s];
                A[s] = alpha * diff + beta * A[s];
            }
        }

        for (int s = 0; s < j; ++s)
            A_store[j][s] = A[s];
    }
}

// ============================================================
// primitive_control_kernel — F-free implementation
//
// calD[j][s] = Pi*D[j][s] + DT * sum_u F[j][u][s]^T * D[j][u]
//
// Decomposed using F[j][u][s] = border + sum_k R[k][u]*A_store[k][s]^T:
//
// For s < j:
//   border_gt = g * sum_{u=s+1}^{j} R[u][s] * D[j][u](obs_index)
//   border_lt = g * e_i * sum_{u<s} dot(R[s][u], D[j][u])
//   interior  = sum_{k=s+1}^{j} A_store[k][s] * partial_d_k
//     where partial_d_k = sum_{u<k} dot(R[k][u], D[j][u])
//
// For s = j:
//   calD[j][j] = Pi*D[j][j] + DT*g*e_i * sum_{u<j} dot(R[j][u], D[j][u])
// ============================================================
static void primitive_control_kernel(
    const Kernel2D& D, const Kernel2D& R, const Kernel2D& A_store,
    double obs_gain_val, int obs_index, const Mat3& Pi, Kernel2D& calD) {

    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain_val;

    for (int j = 0; j < N; ++j) {
        int m = j + 1;

        if (j == 0) {
            calD[0][0] = Pi * D[0][0];
            for (int s = 1; s < N; ++s) calD[0][s].setZero();
            continue;
        }

        // Precompute partial_d_k = sum_{u=0}^{k-1} dot(R[k][u], D[j][u])
        // for k = 0..j.  partial_d_k[s] also equals the border_lt sum,
        // so we reuse it below instead of recomputing.
        std::array<double, N> partial_d_k;
        partial_d_k[0] = 0.0;
        for (int k = 1; k <= j; ++k) {
            double acc = 0.0;
            for (int u = 0; u < k; ++u)
                acc += R[k][u].dot(D[j][u]);
            partial_d_k[k] = acc;
        }

        // Precompute D[j][u](obs_index) for u = 0..j
        std::array<double, N> D_obs;
        for (int u = 0; u <= j; ++u)
            D_obs[u] = D[j][u](obs_index);

        for (int s = 0; s < j; ++s) {
            Vec3 acc = Pi * D[j][s];

            // border_gt + interior merged: both iterate k = s+1..j
            Vec3 bg_vec = Vec3::Zero();
            Vec3 int_acc = Vec3::Zero();
            for (int k = s + 1; k <= j; ++k) {
                bg_vec += D_obs[k] * R[k][s];
                int_acc += partial_d_k[k] * A_store[k][s];
            }
            acc += DT * g * bg_vec + DT * int_acc;

            // border_lt: reuse partial_d_k[s] instead of recomputing
            acc += (DT * g * partial_d_k[s]) * e_i;

            calD[j][s] = acc;
        }

        // s = j: reuse partial_d_k[j]
        calD[j][j] = Pi * D[j][j] + (DT * g * partial_d_k[j]) * e_i;

        for (int s = m; s < N; ++s)
            calD[j][s].setZero();
    }
}

// ============================================================
// forward_environment — F-free
// ============================================================
void forward_environment(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env) {

    Kernel2D calD1, calD2;
    for (int j = 0; j < N; ++j) {
        int m = j + 1;
        for (int s = 0; s < m; ++s) {
            calD1[j][s] = Pi_1 * D1[j][s];
            calD2[j][s] = Pi_2 * D2[j][s];
        }
        for (int s = m; s < N; ++s) {
            calD1[j][s].setZero();
            calD2[j][s].setZero();
        }
    }

    Kernel2D X;
    state_kernel_from_calD(calD1, calD2, X);

    // Write filter results directly into env to avoid 6 × 38KB copies
    Kernel2D calD1_new, calD2_new, X_new;

    for (int it = 0; it < inner_iters; ++it) {
        #pragma omp parallel sections
        {
            #pragma omp section
            compute_filter_kernels(X, Pi_1, obs_idx_1, obs_gain1,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   env.R1, env.A_store1);
            #pragma omp section
            compute_filter_kernels(X, Pi_2, obs_idx_2, obs_gain2,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   env.R2, env.A_store2);
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            primitive_control_kernel(D1, env.R1, env.A_store1, obs_gain1, obs_idx_1, Pi_1, calD1_new);
            #pragma omp section
            primitive_control_kernel(D2, env.R2, env.A_store2, obs_gain2, obs_idx_2, Pi_2, calD2_new);
        }

        state_kernel_from_calD(calD1_new, calD2_new, X_new);

        for (int t = 0; t < N; ++t)
            for (int s = 0; s <= t; ++s)
                X[t][s] = 0.6 * X_new[t][s] + 0.4 * X[t][s];
    }

    env.X = X;
    env.calD1 = calD1_new; env.calD2 = calD2_new;
    env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
    env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
}

// ============================================================
// backward_kernels — ping-pong version (no Kernel3D allocation)
//
// Uses two N×N Mat3 slices (~115KB each, 230KB total) instead of
// the 36MB Kernel3D. Only the previous time-step's Hk slice is
// needed at each backward step.
// ============================================================
using HkSlice = std::array<std::array<Mat3, N>, N>;

void backward_kernels(const Kernel2D& X, const Kernel2D& Rk,
                      const Kernel2D& Dk,
                      const std::array<double, N>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx) {
    for (auto& row : Hx)
        for (auto& v : row)
            v.setZero();

    for (int s = 0; s < N; ++s)
        Hx[N - 1][s] = -terminal_state_weight * X[N - 1][s];

    // Ping-pong buffers: cur holds Hk[t_idx], nxt receives Hk[j]
    HkSlice buf0, buf1;
    for (int z = 0; z < N; ++z)
        for (int r = 0; r < N; ++r)
            buf0[z][r].setZero();

    HkSlice* cur = &buf0;
    HkSlice* nxt = &buf1;

    for (int j = N - 2; j >= 0; --j) {
        int t_idx = j + 1;
        double Pk = prec_k[t_idx];
        int m = j + 1;
        int mz = t_idx + 1;

        std::array<Vec3, N> W;
        for (int r = 0; r < m; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z < mz; ++z)
                acc += (*cur)[z][r].transpose() * Rk[t_idx][z];
            W[r] = DT * Pk * acc;
        }

        for (int r = 0; r < m; ++r)
            Hx[j][r] = Hx[t_idx][r] + DT * (X[t_idx][r] + W[r]);

        for (int z = 0; z < m; ++z) {
            for (int r = 0; r < m; ++r) {
                Mat3 first = Dk[t_idx][r] * Hx[t_idx][z].transpose();
                Mat3 ck = X[t_idx][z] * W[r].transpose();
                (*nxt)[z][r] = (*cur)[z][r] + DT * (first - ck);
            }
        }

        std::swap(cur, nxt);
    }
}

// Legacy version that also fills Hk (for figure output)
void backward_kernels(const Kernel2D& X, const Kernel2D& Rk,
                      const Kernel2D& Dk,
                      const std::array<double, N>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel3D& Hk) {
    for (auto& row : Hx)
        for (auto& v : row)
            v.setZero();

    for (int s = 0; s < N; ++s)
        Hx[N - 1][s] = -terminal_state_weight * X[N - 1][s];

    for (int z = 0; z < N; ++z)
        for (int r = 0; r < N; ++r)
            Hk[N - 1][z][r].setZero();

    for (int j = N - 2; j >= 0; --j) {
        int t_idx = j + 1;
        double Pk = prec_k[t_idx];
        int m = j + 1;
        int mz = t_idx + 1;

        std::array<Vec3, N> W;
        for (int r = 0; r < m; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z < mz; ++z)
                acc += Hk[t_idx][z][r].transpose() * Rk[t_idx][z];
            W[r] = DT * Pk * acc;
        }

        for (int r = 0; r < m; ++r)
            Hx[j][r] = Hx[t_idx][r] + DT * (X[t_idx][r] + W[r]);

        for (int z = 0; z < m; ++z) {
            for (int r = 0; r < m; ++r) {
                Mat3 first = Dk[t_idx][r] * Hx[t_idx][z].transpose();
                Mat3 ck = X[t_idx][z] * W[r].transpose();
                Hk[j][z][r] = Hk[t_idx][z][r] + DT * (first - ck);
            }
        }
    }
}

// ============================================================
// backward_bar_adjoints
// ============================================================
BackwardBarResult backward_bar_adjoints(
    const Kernel2D& X, const Kernel2D& Rk, const Kernel2D& Dk,
    const std::array<double, N>& barX, double b,
    const std::array<double, N>& prec_k, double terminal_weight) {

    BackwardBarResult res;
    res.barHx.fill(0.0);
    for (auto& row : res.barHk)
        for (auto& v : row)
            v.setZero();

    res.barHx[N - 1] = terminal_weight * (barX[N - 1] - b);

    for (int j = N - 2; j >= 0; --j) {
        int t_idx = j + 1;
        double Pk = prec_k[t_idx];
        int mz = t_idx + 1;
        int m_u = j + 1;

        double I_val = 0.0;
        for (int z = 0; z < mz; ++z)
            I_val += Rk[t_idx][z].dot(res.barHk[t_idx][z]);
        I_val *= DT * Pk;

        res.barHx[j] = res.barHx[t_idx] + DT * ((barX[t_idx] - b) + I_val);

        for (int s = 0; s < m_u; ++s)
            res.barHk[j][s] = res.barHk[t_idx][s] +
                DT * (Dk[t_idx][s] * res.barHx[t_idx] - X[t_idx][s] * I_val);
    }

    return res;
}

// ============================================================
// solve_bar_equilibrium
// ============================================================
BarSolution solve_bar_equilibrium(
    const EnvironmentResult& env, const Kernel2D& D1, const Kernel2D& D2,
    double prec1, double prec2,
    int max_iters, double relax, double tol, bool /*verbose*/) {

    BarSolution sol;
    std::array<double, N> barD1, barD2;
    barD1.fill(0.0);
    barD2.fill(0.0);

    auto prec1_arr = make_constant_prec(prec1);
    auto prec2_arr = make_constant_prec(prec2);

    double last_err = 1e30;

    for (int it = 1; it <= max_iters; ++it) {
        std::array<double, N> barX;
        barX[0] = X0;
        for (int j = 0; j < N - 1; ++j)
            barX[j + 1] = barX[j] + DT * (barD1[j] + barD2[j]);

        BackwardBarResult bba1, bba2;
        #pragma omp parallel sections
        {
            #pragma omp section
            bba1 = backward_bar_adjoints(env.X, env.R2, D2, barX, g_b1, prec2_arr, 0.0);
            #pragma omp section
            bba2 = backward_bar_adjoints(env.X, env.R1, D1, barX, g_b2, prec1_arr, 0.0);
        }

        std::array<double, N> barD1_new, barD2_new;
        for (int j = 0; j < N; ++j) {
            barD1_new[j] = -(1.0 / RHO) * bba1.barHx[j];
            barD2_new[j] = -(1.0 / RHO) * bba2.barHx[j];
        }

        double norm1_new = 0.0, norm1_diff = 0.0;
        double norm2_new = 0.0, norm2_diff = 0.0;
        for (int j = 0; j < N; ++j) {
            norm1_diff += (barD1_new[j] - barD1[j]) * (barD1_new[j] - barD1[j]);
            norm1_new += barD1_new[j] * barD1_new[j];
            norm2_diff += (barD2_new[j] - barD2[j]) * (barD2_new[j] - barD2[j]);
            norm2_new += barD2_new[j] * barD2_new[j];
        }
        double err1 = std::sqrt(norm1_diff) / std::max(1.0, std::sqrt(norm1_new));
        double err2 = std::sqrt(norm2_diff) / std::max(1.0, std::sqrt(norm2_new));
        last_err = std::max(err1, err2);

        for (int j = 0; j < N; ++j) {
            barD1[j] = (1.0 - relax) * barD1[j] + relax * barD1_new[j];
            barD2[j] = (1.0 - relax) * barD2[j] + relax * barD2_new[j];
        }

        if (last_err < tol) break;
    }

    sol.barX[0] = X0;
    for (int j = 0; j < N - 1; ++j)
        sol.barX[j + 1] = sol.barX[j] + DT * (barD1[j] + barD2[j]);
    sol.barD1 = barD1;
    sol.barD2 = barD2;
    sol.bar_residual = last_err;
    return sol;
}

// ============================================================
// Anderson-accelerated inner product over causal triangle
// ============================================================
static double kernel_dot(const Kernel2D& a1, const Kernel2D& a2,
                         const Kernel2D& b1, const Kernel2D& b2) {
    double s = 0.0;
    for (int t = 0; t < N; ++t)
        for (int s2 = 0; s2 <= t; ++s2)
            s += a1[t][s2].dot(b1[t][s2]) + a2[t][s2].dot(b2[t][s2]);
    return s;
}

// ============================================================
// solve_equilibrium — Anderson-accelerated Picard iteration
//
// Anderson(m) acceleration uses the last m residuals to find
// optimal mixing coefficients that extrapolate toward the fixed
// point, typically reducing iteration count by 2-3x.
// ============================================================
EquilibriumResult solve_equilibrium(
    double p1_val, double p2_val, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {

    Kernel2D D1, D2;
    for (auto& row : D1) for (auto& v : row) v.setZero();
    for (auto& row : D2) for (auto& v : row) v.setZero();

    std::vector<double> residuals;
    double P1 = p1_val * p1_val;
    double P2 = p2_val * p2_val;

    auto prec1 = make_constant_prec(P1);
    auto prec2 = make_constant_prec(P2);

    EnvironmentResult env;

    // Anderson acceleration state
    constexpr int AA_M = 5;
    constexpr double AA_REG = 1e-12;
    struct AAEntry { Kernel2D f1, f2, g1, g2; };
    std::vector<AAEntry> aa_hist(AA_M);
    int aa_stored = 0;

    for (int it = 1; it <= MAX_PICARD_ITERS; ++it) {
        forward_environment(D1, D2, p1_val, p2_val,
                            FORWARD_INNER_ITERS,
                            Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);

        Kernel2D Hx1, Hx2;
        #pragma omp parallel sections
        {
            #pragma omp section
            backward_kernels(env.X, env.R2, D2, prec2, 0.0, Hx1);
            #pragma omp section
            backward_kernels(env.X, env.R1, D1, prec1, 0.0, Hx2);
        }

        // Compute Picard image G and residual F = G - D
        constexpr double neg_inv_rho = -(1.0 / RHO);
        Kernel2D G1, G2, F1, F2;
        double norm_fn = 0.0, norm_gn = 0.0;
        for (int t = 0; t < N; ++t)
            for (int s = 0; s <= t; ++s) {
                G1[t][s] = neg_inv_rho * Hx1[t][s];
                G2[t][s] = neg_inv_rho * Hx2[t][s];
                F1[t][s] = G1[t][s] - D1[t][s];
                F2[t][s] = G2[t][s] - D2[t][s];
                norm_fn += F1[t][s].squaredNorm() + F2[t][s].squaredNorm();
                norm_gn += G1[t][s].squaredNorm() + G2[t][s].squaredNorm();
            }

        double err = std::sqrt(norm_fn) / std::max(1.0, std::sqrt(norm_gn));
        residuals.push_back(err);

        if (!std::isfinite(err)) break;

        if (verbose && (it <= 5 || it % 50 == 0))
            std::cout << "  it=" << it << "  resid=" << err << std::endl;

        if (err < PICARD_TOL) {
            if (verbose)
                std::cout << "  Converged at iteration " << it << std::endl;
            break;
        }

        // Anderson acceleration
        int m = std::min(aa_stored, AA_M);
        bool did_anderson = false;

        if (m >= 1) {
            // ΔF_i = F_curr - f_hist[i], for i = 0..m-1 (most recent first)
            // Gram matrix H[i][j] = <ΔF_i, ΔF_j>, rhs[i] = <ΔF_i, F_curr>
            // Using: <ΔF_i, ΔF_j> = <F,F> - <F,f_j> - <f_i,F> + <f_i,f_j>
            double FF = norm_fn;
            Eigen::VectorXd Ffi(m);
            for (int i = 0; i < m; ++i) {
                int idx = (aa_stored - 1 - i + AA_M) % AA_M;
                Ffi(i) = kernel_dot(F1, F2, aa_hist[idx].f1, aa_hist[idx].f2);
            }

            Eigen::MatrixXd H(m, m);
            Eigen::VectorXd rhs(m);
            for (int i = 0; i < m; ++i) {
                rhs(i) = FF - Ffi(i);
                for (int j = 0; j <= i; ++j) {
                    int ii = (aa_stored - 1 - i + AA_M) % AA_M;
                    int jj = (aa_stored - 1 - j + AA_M) % AA_M;
                    double fifj = kernel_dot(aa_hist[ii].f1, aa_hist[ii].f2,
                                             aa_hist[jj].f1, aa_hist[jj].f2);
                    H(i, j) = FF - Ffi(i) - Ffi(j) + fifj;
                    H(j, i) = H(i, j);
                }
            }

            // Tikhonov regularization
            for (int i = 0; i < m; ++i)
                H(i, i) += AA_REG * (1.0 + H(i, i));

            Eigen::VectorXd gamma = H.ldlt().solve(rhs);

            // x_{k+1} = G - Σ γ_i * (G - g_hist[i])
            //         = (1 - Σγ) * G + Σ γ_i * g_hist[i]
            Kernel2D D1_aa = G1, D2_aa = G2;
            for (int i = 0; i < m; ++i) {
                int idx = (aa_stored - 1 - i + AA_M) % AA_M;
                double gi = gamma(i);
                for (int t = 0; t < N; ++t)
                    for (int s = 0; s <= t; ++s) {
                        D1_aa[t][s] -= gi * (G1[t][s] - aa_hist[idx].g1[t][s]);
                        D2_aa[t][s] -= gi * (G2[t][s] - aa_hist[idx].g2[t][s]);
                    }
            }

            // Damped Anderson: blend with simple relaxation for stability
            // x_{k+1} = (1-β)*D + β*D_aa, with β tuned for the problem
            constexpr double AA_BETA = 0.6;
            for (int t = 0; t < N; ++t)
                for (int s = 0; s <= t; ++s) {
                    D1[t][s] = (1.0 - AA_BETA) * D1[t][s] + AA_BETA * D1_aa[t][s];
                    D2[t][s] = (1.0 - AA_BETA) * D2[t][s] + AA_BETA * D2_aa[t][s];
                }
            did_anderson = true;
        }

        if (!did_anderson) {
            // Simple relaxation for first iteration
            for (int t = 0; t < N; ++t)
                for (int s = 0; s <= t; ++s) {
                    D1[t][s] += PICARD_RELAX * F1[t][s];
                    D2[t][s] += PICARD_RELAX * F2[t][s];
                }
        }

        // Store in ring buffer
        int store_idx = aa_stored % AA_M;
        aa_hist[store_idx].f1 = F1;
        aa_hist[store_idx].f2 = F2;
        aa_hist[store_idx].g1 = G1;
        aa_hist[store_idx].g2 = G2;
        aa_stored++;
    }

    // env is already computed for the converged D1/D2 — the convergence
    // check breaks BEFORE the Anderson update modifies D, so no final
    // forward_environment is needed.

    return {D1, D2, std::move(env), residuals};
}

// ============================================================
// compute_costs_general
// ============================================================
CostPair compute_costs_general(const EnvironmentResult& env,
                               const BarSolution& bar_sol,
                               double r_val, double b1_val, double b2_val) {
    double J1 = 0.0, J2 = 0.0;
    for (int j = 0; j < N; ++j) {
        double var_X = 0.0;
        for (int s = 0; s < j; ++s)
            var_X += DT * env.X[j][s].squaredNorm();

        double var_D1 = 0.0, var_D2 = 0.0;
        for (int s = 0; s < j; ++s) {
            var_D1 += DT * env.calD1[j][s].squaredNorm();
            var_D2 += DT * env.calD2[j][s].squaredNorm();
        }

        double dx1 = bar_sol.barX[j] - b1_val;
        double dx2 = bar_sol.barX[j] - b2_val;

        J1 += DT * (dx1 * dx1 + var_X + r_val * bar_sol.barD1[j] * bar_sol.barD1[j] + r_val * var_D1);
        J2 += DT * (dx2 * dx2 + var_X + r_val * bar_sol.barD2[j] * bar_sol.barD2[j] + r_val * var_D2);
    }
    return {J1, J2};
}

// ============================================================
// materialize_F — builds F[j][u][s] from R + A_store for figure output
//
// F[j][u][s] for u,s < j:
//   border + sum_{k=max(u,s)+1}^{j} R[k][u] * A_store[k][s]^T
//
// Border at step max(u,s):
//   u > s: F[u][u][s] = g*e_i*R[u][s]^T
//   u < s: F[s][u][s] = g*R[s][u]*e_i^T
//   u = s: 0
//
// Border row/col at step j:
//   F[j][u][j] = g*R[j][u]*e_i^T  (u < j)
//   F[j][j][s] = g*e_i*R[j][s]^T  (s < j)
//   F[j][j][j] = 0
// ============================================================
void materialize_F(const Kernel2D& R, const Kernel2D& A_store,
                   double obs_gain, int obs_index, Kernel3D& F) {
    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain;

    // Build F incrementally: F[0], F[1], ..., F[N-1]
    F[0][0][0].setZero();

    for (int j = 1; j < N; ++j) {
        // Set borders
        for (int u = 0; u < j; ++u) {
            F[j][u][j] = g * R[j][u] * e_i.transpose();
            F[j][j][u] = g * e_i * R[j][u].transpose();
        }
        F[j][j][j].setZero();

        // Interior: F[j][u][s] = F[j-1][u][s] + R[j][u] * A_store[j][s]^T
        for (int u = 0; u < j; ++u)
            for (int s = 0; s < j; ++s)
                F[j][u][s] = F[j - 1][u][s] + R[j][u] * A_store[j][s].transpose();
    }
}
