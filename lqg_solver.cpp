// ============================================================
// Decentralized LQG noise-state game solver — optimized implementation
// Uses direct 3x3 block operations instead of flat dynamic matrices.
// ============================================================

#include "lqg_solver.h"
#include <algorithm>
#include <cstring>

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
// Q stored as block matrix: Q_blocks[u][s] is a 3x3 matrix
// Q_blocks[u][s] corresponds to Q_flat[u*dw..(u+1)*dw, s*dw..(s+1)*dw]
// ============================================================
using QBlocks = std::array<std::array<Mat3, N>, N>;

// ============================================================
// compute_filter_kernels — optimized with direct block ops
// ============================================================
void compute_filter_kernels(const Kernel2D& X, const Mat3& Pi, int obs_index,
                            const std::array<double, N>& obs_gain,
                            int filter_iters, double relax,
                            Kernel2D& R, Eigen::MatrixXd& /*Q_flat_unused*/,
                            Kernel3D& F, Kernel2D& Xhat) {
    // Zero-init only the parts we'll use
    // R[j][s] for s <= j, Xhat[j][s] for s <= j, F[j][u][s] for u,s <= j

    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;

    // Q as block matrix — only upper-left (j+1)x(j+1) block used at step j
    // Allocated once, accumulated across j
    QBlocks Q_blocks;
    for (auto& row : Q_blocks)
        for (auto& m : row)
            m.setZero();

    // Pre-allocate scratch arrays for Gamma computation (avoid heap allocation)
    std::array<Vec3, N> total_sum_buf, gamma_a_buf, inclusive_buf;
    std::array<double, N> coeff_a_buf, b_scalar_buf;
    // cum_diag[s] = cumulative weighted R[z][s] evaluated at z=s (for inclusive)
    // We don't need the full cum[z][s] matrix — we only need cum[s][s] and total_sum[s]
    // gamma_a[s] = dt * (total_sum[s] - inclusive[s])
    // where inclusive[s] = cum[s][s] for s < j, total_sum[s] for s >= j.
    // cum[s][s] = sum_{z=0}^{s} coeff_a[z] * R[z][s]
    // So we can compute this incrementally without storing the full matrix.

    for (int j = 0; j < N; ++j) {
        double g = obs_gain[j];
        double prec = g * g;
        int m = j + 1;

        // --- Q_corr[s] = dt * sum_u Q_blocks[u][s]^T * X[j][u], for s in [0,m) ---
        // Q_corr replaces: dt * (Xj_flat^T @ Q_flat)^T
        std::array<Vec3, N> Q_corr;
        for (int s = 0; s < m; ++s) {
            Vec3 acc = Vec3::Zero();
            for (int u = 0; u < m; ++u)
                acc += Q_blocks[u][s].transpose() * X[j][u];
            Q_corr[s] = DT * acc;
        }

        // --- Gamma computation ---
        // gamma_a[s] = dt * (total_sum[s] - inclusive[s])
        // gamma_b[s] = dt * b_scalar[s] * obs_gain[s] * e_i
        // Gamma[s] = gamma_a[s] + gamma_b[s]
        std::array<Vec3, N>& Gamma = gamma_a_buf; // reuse buffer for final result

        if (j == 0) {
            Gamma[0].setZero();
        } else {
            // coeff_a[z] = X[j][z](obs_index) * obs_gain[z]
            for (int z = 0; z < j; ++z)
                coeff_a_buf[z] = X[j][z](obs_index) * obs_gain[z];

            // Compute total_sum[s] = sum_{z=0}^{j-1} coeff_a[z] * R[z][s]
            for (int s = 0; s < m; ++s)
                total_sum_buf[s].setZero();

            for (int z = 0; z < j; ++z) {
                double ca = coeff_a_buf[z];
                for (int s = 0; s < m; ++s)
                    total_sum_buf[s] += ca * R[z][s];
            }

            // inclusive[s]:
            //   s < j: cum[s][s] = sum_{z=0}^{s} coeff_a[z] * R[z][s]
            //   s >= j: total_sum[s]
            int jj = std::min(j, m);
            // For s < jj, compute partial sum up to z=s
            // We can do this by accumulating per-s
            for (int s = 0; s < jj; ++s) {
                Vec3 acc = Vec3::Zero();
                for (int z = 0; z <= s; ++z)
                    acc += coeff_a_buf[z] * R[z][s];
                inclusive_buf[s] = acc;
            }
            for (int s = j; s < m; ++s)
                inclusive_buf[s] = total_sum_buf[s];

            // gamma_a[s] = dt * (total_sum[s] - inclusive[s])
            for (int s = 0; s < m; ++s)
                gamma_a_buf[s] = DT * (total_sum_buf[s] - inclusive_buf[s]);

            // gamma_b: b_scalar[s], then gamma_b[s] = dt * b_scalar[s] * obs_gain[s] * e_i
            // dot_mat[z][s] = X[j][z] . R[z][s]
            // cum_dot[z][s] = cumsum over z
            // b_scalar[s] = cum_dot[s-1][s] for 1<=s<jj, cum_dot[j-1][s] for s>=j
            // b_scalar[0] = 0

            // Compute b_scalar without storing full dot_mat/cum_dot matrices
            for (int s = 0; s < m; ++s)
                b_scalar_buf[s] = 0.0;

            for (int s = 1; s < jj; ++s) {
                double cum = 0.0;
                for (int z = 0; z < s; ++z) // z < s, so cum_dot[s-1][s] = sum_{z=0}^{s-1}
                    cum += X[j][z].dot(R[z][s]);
                b_scalar_buf[s] = cum;
            }
            if (m > j && j > 0) {
                // For s >= j: b_scalar[s] = cum_dot[j-1][s] = sum_{z=0}^{j-1} X[j][z].dot(R[z][s])
                for (int s = j; s < m; ++s) {
                    double cum = 0.0;
                    for (int z = 0; z < j; ++z)
                        cum += X[j][z].dot(R[z][s]);
                    b_scalar_buf[s] = cum;
                }
            }

            // Gamma[s] = gamma_a[s] + gamma_b[s]
            for (int s = 0; s < m; ++s)
                Gamma[s] += (DT * b_scalar_buf[s] * obs_gain[s]) * e_i;
        }

        // --- R[j][s] = (I - Pi) * X[j][s] - Q_corr[s] - Gamma[s] ---
        Mat3 I_minus_Pi = Mat3::Identity() - Pi;
        for (int s = 0; s < m; ++s)
            R[j][s] = I_minus_Pi * X[j][s] - Q_corr[s] - Gamma[s];
        // Zero out unused entries
        for (int s = m; s < N; ++s)
            R[j][s].setZero();

        // --- Xhat & F computation using direct block operations ---
        if (j == 0) {
            Xhat[j][0] = Pi * X[j][0];
            F[j][0][0].setZero();
        } else {
            // Set new row/column of F[j] (border blocks)
            for (int u = 0; u < j; ++u) {
                F[j][u][j] = g * R[j][u] * e_i.transpose();
                F[j][j][u] = g * e_i * R[j][u].transpose();
            }
            F[j][j][j].setZero();

            // Copy F[j-1] → F[j] for interior block (u,s < j)
            // and track correction delta[u][s] = F[j][u][s] - F[j-1][u][s]
            // Initialize delta = 0, so F[j][u][s] = F[j-1][u][s] initially
            // We reference F[j-1] as base to avoid a separate copy.

            // Precompute Pi * X[j][s] and the contribution from the fixed s=j column
            std::array<Vec3, N> Xhat_const; // constant part across iterations
            for (int s = 0; s < m; ++s) {
                Vec3 acc = Pi * X[j][s];
                // u = j contribution (fixed): F[j][j][s]^T * X[j][j]
                acc += DT * F[j][j][s].transpose() * X[j][j];
                // u < j contribution from F[j-1] (base):
                for (int u = 0; u < j; ++u)
                    acc += DT * F[j - 1][u][s].transpose() * X[j][u];
                Xhat_const[s] = acc;
            }

            // Fixed-point iteration updates delta[u][s] for u,s < j
            // F_eff[u][s] = F[j-1][u][s] + delta[u][s]
            // Xhat[s] = Xhat_const[s] + DT * sum_{u<j} delta[u][s]^T * X[j][u]
            double dt_prec = DT * prec;
            double one_minus_relax = 1.0 - relax;

            // delta starts at zero
            // Store as flat array of Mat3
            std::array<std::array<Mat3, N>, N> delta;
            for (int u = 0; u < j; ++u)
                for (int s = 0; s < j; ++s)
                    delta[u][s].setZero();

            // Precompute DT * X[j][u] for u < j
            std::array<Vec3, N> dtXju;
            for (int u = 0; u < j; ++u)
                dtXju[u] = DT * X[j][u];

            for (int iter = 0; iter < filter_iters; ++iter) {
                // Compute diff[s] = X[j][s] - Xhat[s] for s < j
                // Xhat[s] = Xhat_const[s] + sum_{u<j} delta[u][s]^T * dtXju[u]
                std::array<Vec3, N> diff;
                for (int s = 0; s < j; ++s) {
                    Vec3 delta_contrib = Vec3::Zero();
                    for (int u = 0; u < j; ++u)
                        delta_contrib += delta[u][s].transpose() * dtXju[u];
                    diff[s] = X[j][s] - Xhat_const[s] - delta_contrib;
                }

                // Update delta: new_delta[u][s] = relax * dt_prec * R[j][u] * diff[s]^T
                //                                 + (1-relax) * delta[u][s]
                for (int u = 0; u < j; ++u) {
                    Vec3 Ru_scaled = (relax * dt_prec) * R[j][u];
                    for (int s = 0; s < j; ++s)
                        delta[u][s] = Ru_scaled * diff[s].transpose()
                                      + one_minus_relax * delta[u][s];
                }
            }

            // Write final F[j][u][s] = F[j-1][u][s] + delta[u][s]
            for (int u = 0; u < j; ++u)
                for (int s = 0; s < j; ++s)
                    F[j][u][s] = F[j - 1][u][s] + delta[u][s];

            // Final Xhat
            for (int s = 0; s < m; ++s) {
                Vec3 acc = Xhat_const[s];
                for (int u = 0; u < j; ++u)
                    acc += delta[u][s].transpose() * dtXju[u];
                Xhat[j][s] = acc;
            }
        }

        // --- Update Q_blocks: Q[u][s] += dt * prec * R[j][u] * R[j][s]^T ---
        if (j < N - 1) {
            double coeff = DT * prec;
            for (int u = 0; u < m; ++u) {
                Vec3 Ru_scaled = coeff * R[j][u];
                for (int s = 0; s < m; ++s)
                    Q_blocks[u][s] += Ru_scaled * R[j][s].transpose();
            }
        }
    }
}

// ============================================================
// primitive_control_kernel — direct block operations
// calD[j][s] = Pi * D[j][s] + dt * sum_u F[j][u][s]^T * D[j][u]
// ============================================================
void primitive_control_kernel(const Kernel2D& D, const Kernel3D& F,
                              const Mat3& Pi, Kernel2D& calD) {
    for (int j = 0; j < N; ++j) {
        int m = j + 1;
        for (int s = 0; s < m; ++s) {
            Vec3 acc = Pi * D[j][s];
            for (int u = 0; u < m; ++u)
                acc += DT * F[j][u][s].transpose() * D[j][u];
            calD[j][s] = acc;
        }
        for (int s = m; s < N; ++s)
            calD[j][s].setZero();
    }
}

// ============================================================
// forward_environment
// ============================================================
void forward_environment(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env) {

    // Initial calD from just Pi projection
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

    std::array<double, N> gain1, gain2;
    gain1.fill(obs_gain1);
    gain2.fill(obs_gain2);

    Kernel2D R1, R2, Xhat1, Xhat2;
    Eigen::MatrixXd Q_dummy; // not used in optimized path
    // Reuse F1, F2 from env to avoid reallocation
    Kernel3D& F1 = env.F1;
    Kernel3D& F2 = env.F2;

    for (int it = 0; it < inner_iters; ++it) {
        compute_filter_kernels(X, Pi_1, obs_idx_1, gain1,
                               FILTER_INNER_ITERS, FILTER_RELAX,
                               R1, Q_dummy, F1, Xhat1);
        compute_filter_kernels(X, Pi_2, obs_idx_2, gain2,
                               FILTER_INNER_ITERS, FILTER_RELAX,
                               R2, Q_dummy, F2, Xhat2);

        Kernel2D calD1_new, calD2_new;
        primitive_control_kernel(D1, F1, Pi_1, calD1_new);
        primitive_control_kernel(D2, F2, Pi_2, calD2_new);

        Kernel2D X_new;
        state_kernel_from_calD(calD1_new, calD2_new, X_new);

        // Relaxation
        for (int t = 0; t < N; ++t)
            for (int s = 0; s < N; ++s)
                X[t][s] = 0.6 * X_new[t][s] + 0.4 * X[t][s];

        calD1 = calD1_new;
        calD2 = calD2_new;
    }

    env.X = X;
    env.R1 = R1; env.R2 = R2;
    // F1, F2 already written in-place via references
    env.Xhat1 = Xhat1; env.Xhat2 = Xhat2;
    env.calD1 = calD1; env.calD2 = calD2;
}

// ============================================================
// backward_kernels — only zero used portions
// ============================================================
void backward_kernels(const Kernel2D& X, const Kernel2D& Rk,
                      const Kernel2D& Dk,
                      const std::array<double, N>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel3D& Hk) {
    // Zero init Hx
    for (auto& row : Hx)
        for (auto& v : row)
            v.setZero();

    // Terminal condition
    for (int s = 0; s < N; ++s)
        Hx[N - 1][s] = -terminal_state_weight * X[N - 1][s];

    // Zero Hk[N-1] (only used portion)
    for (int z = 0; z < N; ++z)
        for (int r = 0; r < N; ++r)
            Hk[N - 1][z][r].setZero();

    for (int j = N - 2; j >= 0; --j) {
        int t_idx = j + 1;
        double Pk = prec_k[t_idx];
        int m = j + 1;
        int mz = t_idx + 1;

        // Compute coupling/W for each r (reused in Hx and Hk)
        // W[r] = dt * Pk * sum_z Hk[t_idx][z][r]^T * Rk[t_idx][z]
        std::array<Vec3, N> W;
        for (int r = 0; r < m; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z < mz; ++z)
                acc += Hk[t_idx][z][r].transpose() * Rk[t_idx][z];
            W[r] = DT * Pk * acc;
        }

        // Hx[j][r] = Hx[t_idx][r] + dt * (X[t_idx][r] + W[r])
        for (int r = 0; r < m; ++r)
            Hx[j][r] = Hx[t_idx][r] + DT * (X[t_idx][r] + W[r]);

        // Hk[j][z][r] = Hk[t_idx][z][r] + dt * (Dk[t_idx][r] * Hx[t_idx][z]^T - X[t_idx][z] * W[r]^T)
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

        auto bba1 = backward_bar_adjoints(env.X, env.R2, D2, barX, g_b1, prec2_arr, 0.0);
        auto bba2 = backward_bar_adjoints(env.X, env.R1, D1, barX, g_b2, prec1_arr, 0.0);

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
// solve_equilibrium
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

    // Pre-allocate to avoid repeated 36MB heap alloc/free per iteration
    EnvironmentResult env;
    Kernel3D Hk1, Hk2;

    for (int it = 1; it <= MAX_PICARD_ITERS; ++it) {
        forward_environment(D1, D2, p1_val, p2_val,
                            FORWARD_INNER_ITERS,
                            Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);

        Kernel2D Hx1, Hx2;
        backward_kernels(env.X, env.R2, D2, prec2, 0.0, Hx1, Hk1);
        backward_kernels(env.X, env.R1, D1, prec1, 0.0, Hx2, Hk2);

        Kernel2D D1_new, D2_new;
        for (int t = 0; t < N; ++t)
            for (int s = 0; s < N; ++s) {
                D1_new[t][s] = -(1.0 / RHO) * Hx1[t][s];
                D2_new[t][s] = -(1.0 / RHO) * Hx2[t][s];
            }

        // Compute relative error
        double norm_d1n = 0.0, norm_d1d = 0.0;
        double norm_d2n = 0.0, norm_d2d = 0.0;
        for (int t = 0; t < N; ++t)
            for (int s = 0; s < N; ++s) {
                norm_d1d += (D1_new[t][s] - D1[t][s]).squaredNorm();
                norm_d1n += D1_new[t][s].squaredNorm();
                norm_d2d += (D2_new[t][s] - D2[t][s]).squaredNorm();
                norm_d2n += D2_new[t][s].squaredNorm();
            }
        double err = std::max(
            std::sqrt(norm_d1d) / std::max(1.0, std::sqrt(norm_d1n)),
            std::sqrt(norm_d2d) / std::max(1.0, std::sqrt(norm_d2n)));
        residuals.push_back(err);

        if (!std::isfinite(err)) break;

        for (int t = 0; t < N; ++t)
            for (int s = 0; s < N; ++s) {
                D1[t][s] = (1.0 - PICARD_RELAX) * D1[t][s] +
                           PICARD_RELAX * D1_new[t][s];
                D2[t][s] = (1.0 - PICARD_RELAX) * D2[t][s] +
                           PICARD_RELAX * D2_new[t][s];
            }

        if (verbose && (it <= 5 || it % 50 == 0))
            std::cout << "  it=" << it << "  resid=" << err << std::endl;

        if (err < PICARD_TOL) {
            if (verbose)
                std::cout << "  Converged at iteration " << it << std::endl;
            break;
        }
    }

    forward_environment(D1, D2, p1_val, p2_val,
                        FORWARD_INNER_ITERS,
                        Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);

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
