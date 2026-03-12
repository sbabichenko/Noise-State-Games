// ============================================================
// Decentralized LQG noise-state game solver — optimized implementation
//
// Key optimizations over naive port:
// 1. Direct 3x3 block ops instead of dynamic Eigen flat matrices
// 2. Rank-1 factorization of filter delta: delta[u][s] = R[j][u]*A[s]^T
//    reduces filter iteration from O(j^2) to O(j) per step
// 3. Factored Q_corr from R history (scalar*Vec3 vs Mat3^T*Vec3)
// 4. Pre-allocated Kernel3D to avoid heap churn
// 5. Simplified Gamma computation
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
// compute_filter_kernels — fully optimized
//
// Major optimization: the filter fixed-point iteration maintains
//   delta[u][s] = R[j][u] * A[s]^T   (rank-1 in u)
// so we only track A[s] (Vec3 per s) instead of j*j Mat3 blocks.
// This reduces the filter loop from O(j^2 * iters) to O(j * iters).
//
// Q_corr uses factored form from R history:
//   Q_corr[s] = DT^2 * prec * sum_{k<j} c_k * R[k][s]
//   where c_k = sum_{u<=k} dot(R[k][u], X[j][u])
// ============================================================
void compute_filter_kernels(const Kernel2D& X, const Mat3& Pi, int obs_index,
                            const std::array<double, N>& obs_gain,
                            int filter_iters, double relax,
                            Kernel2D& R, Eigen::MatrixXd& /*Q_flat_unused*/,
                            Kernel3D& F, Kernel2D& Xhat) {
    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    Mat3 I_minus_Pi = Mat3::Identity() - Pi;

    // Filter factorization storage: A_store[k][s] = Vec3
    // A_store[k] is the A vector from the filter iteration at step k.
    // Used for Xhat_const computation and primitive_control_kernel via F.
    // F[j][u][s] = F[j-1][u][s] + R[j][u] * A_j[s]^T  for u,s < j
    Kernel2D A_store; // reuse Kernel2D type: A_store[k][s] for s < k

    for (int j = 0; j < N; ++j) {
        double g = obs_gain[j];
        double prec = g * g;
        int m = j + 1;

        // --- Q_corr from R history (factored form) ---
        // Q_corr[s] = DT^2 * prec * sum_{k=s}^{j-1} c_k * R[k][s]
        // where c_k = sum_{u=0}^{k} dot(R[k][u], X[j][u])
        std::array<Vec3, N> Q_corr;
        if (j == 0) {
            Q_corr[0].setZero();
        } else {
            // Compute c_k for k = 0..j-1
            std::array<double, N> c_k;
            for (int k = 0; k < j; ++k) {
                double acc = 0.0;
                for (int u = 0; u <= k; ++u)
                    acc += R[k][u].dot(X[j][u]);
                c_k[k] = acc;
            }
            double dt2_prec = DT * DT * prec;
            for (int s = 0; s < m; ++s) {
                Vec3 acc = Vec3::Zero();
                int k_start = std::max(s, 0);
                for (int k = k_start; k < j; ++k)
                    acc += c_k[k] * R[k][s];
                Q_corr[s] = dt2_prec * acc;
            }
        }

        // --- Gamma computation (simplified) ---
        // gamma_a[s] = dt * sum_{z=s+1}^{j-1} coeff_a[z] * R[z][s]   for s < j
        // gamma_a[s] = 0 for s >= j
        // gamma_b[s] = dt * b_scalar[s] * obs_gain[s] * e_i
        std::array<Vec3, N> Gamma;

        if (j == 0) {
            Gamma[0].setZero();
        } else {
            // coeff_a[z] = X[j][z](obs_index) * obs_gain[z]
            std::array<double, N> coeff_a;
            for (int z = 0; z < j; ++z)
                coeff_a[z] = X[j][z](obs_index) * obs_gain[z];

            // gamma_a: direct sum for z > s
            for (int s = 0; s < j; ++s) {
                Vec3 acc = Vec3::Zero();
                for (int z = s + 1; z < j; ++z)
                    acc += coeff_a[z] * R[z][s];
                Gamma[s] = DT * acc;
            }
            Gamma[j].setZero(); // s = j: gamma_a = 0

            // gamma_b: b_scalar[s] * obs_gain[s] * e_i
            // b_scalar[s] = sum_{z=0}^{min(s,j)-1} dot(X[j][z], R[z][s])
            for (int s = 1; s < j; ++s) {
                double cum = 0.0;
                for (int z = 0; z < s; ++z)
                    cum += X[j][z].dot(R[z][s]);
                Gamma[s] += (DT * cum * obs_gain[s]) * e_i;
            }
            // s >= j (only s = j here): b_scalar = sum_{z=0}^{j-1}
            if (j > 0) {
                double cum = 0.0;
                for (int z = 0; z < j; ++z)
                    cum += X[j][z].dot(R[z][j]);
                Gamma[j] += (DT * cum * obs_gain[j]) * e_i;
            }
        }

        // --- R[j][s] = (I - Pi) * X[j][s] - Q_corr[s] - Gamma[s] ---
        for (int s = 0; s < m; ++s)
            R[j][s] = I_minus_Pi * X[j][s] - Q_corr[s] - Gamma[s];
        for (int s = m; s < N; ++s)
            R[j][s].setZero();

        // --- Xhat & F computation ---
        if (j == 0) {
            Xhat[j][0] = Pi * X[j][0];
            F[j][0][0].setZero();
            A_store[0][0].setZero(); // no correction at step 0
        } else {
            // Set border blocks of F[j]
            for (int u = 0; u < j; ++u) {
                F[j][u][j] = g * R[j][u] * e_i.transpose();
                F[j][j][u] = g * e_i * R[j][u].transpose();
            }
            F[j][j][j].setZero();

            // Xhat_const[s] = Pi*X[j][s] + DT * F[j][j][s]^T * X[j][j]
            //                 + DT * sum_{u<j} F[j-1][u][s]^T * X[j][u]
            //
            // F[j-1][u][s] can be expanded using stored A vectors:
            //   F[j-1][u][s] = sum_{k: max(u,s)<k<=j-1} R[k][u] * A_store[k][s]^T
            //                  + border contributions
            // But this recursive expansion is complex. Instead, compute directly
            // using the already-built F[j-1] (which is fully materialized).
            //
            // For s < j: use F[j-1][u][s] (valid, set at step j-1)
            // For s = j: use F[j][u][j] (border, just set above)
            std::array<Vec3, N> Xhat_const;
            for (int s = 0; s < j; ++s) {
                Vec3 acc = Pi * X[j][s];
                acc += DT * F[j][j][s].transpose() * X[j][j];
                for (int u = 0; u < j; ++u)
                    acc += DT * F[j - 1][u][s].transpose() * X[j][u];
                Xhat_const[s] = acc;
            }
            // s = j: use border column F[j][u][j]
            {
                Vec3 acc = Pi * X[j][j];
                // F[j][j][j] = 0, no u=j contribution
                for (int u = 0; u < j; ++u)
                    acc += DT * F[j][u][j].transpose() * X[j][u];
                Xhat_const[j] = acc;
            }

            // --- Rank-1 factorized filter iteration ---
            // delta[u][s] = R[j][u] * A[s]^T
            // sigma = sum_{u<j} dot(R[j][u], DT * X[j][u]) — precomputed scalar
            // diff[s] = X[j][s] - Xhat_const[s] - sigma * A[s]
            // A_new[s] = alpha * diff[s] + beta * A[s]
            //   where alpha = relax * DT * prec, beta = 1 - relax

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

            // Store A for later use (Xhat_const at future steps via F)
            for (int s = 0; s < j; ++s)
                A_store[j][s] = A[s];

            // Write F[j][u][s] = F[j-1][u][s] + R[j][u] * A[s]^T for u,s < j
            for (int u = 0; u < j; ++u)
                for (int s = 0; s < j; ++s)
                    F[j][u][s] = F[j - 1][u][s] + R[j][u] * A[s].transpose();

            // Final Xhat
            for (int s = 0; s < j; ++s)
                Xhat[j][s] = Xhat_const[s] + sigma * A[s];
            Xhat[j][j] = Xhat_const[j]; // delta has no s=j component
        }
    }
}

// ============================================================
// primitive_control_kernel — direct block operations
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
    Eigen::MatrixXd Q_dummy;
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

        for (int t = 0; t < N; ++t)
            for (int s = 0; s < N; ++s)
                X[t][s] = 0.6 * X_new[t][s] + 0.4 * X[t][s];

        calD1 = calD1_new;
        calD2 = calD2_new;
    }

    env.X = X;
    env.R1 = R1; env.R2 = R2;
    env.Xhat1 = Xhat1; env.Xhat2 = Xhat2;
    env.calD1 = calD1; env.calD2 = calD2;
}

// ============================================================
// backward_kernels
// ============================================================
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
