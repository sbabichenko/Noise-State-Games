// ============================================================
// Decentralized LQG noise-state game solver — implementation
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
// Helper: _to_flat / _from_flat for filter kernels
// ============================================================
// F4d[u][s] is a D_W x D_W matrix, for u,s in [0,m)
// Flat layout: (m*D_W) x (m*D_W) with transpose(0,2,1,3) indexing
// to_flat: F4d[t_idx][u][s](a,b) → flat[u*dw+a, s*dw+b]
// Python: F[:m,:m].transpose(0,2,1,3).reshape(m*dw, m*dw)
static inline void to_flat(const Kernel3D& F4d, int t_idx, int m,
                           Eigen::MatrixXd& out) {
    int sz = m * D_W;
    out.setZero(sz, sz);
    for (int u = 0; u < m; ++u)
        for (int s = 0; s < m; ++s)
            for (int a = 0; a < D_W; ++a)
                for (int b = 0; b < D_W; ++b)
                    out(u * D_W + a, s * D_W + b) = F4d[t_idx][u][s](a, b);
}

static void from_flat(const Eigen::MatrixXd& flat, int m,
                      Kernel3D& F4d, int t_idx) {
    for (int u = 0; u < m; ++u)
        for (int s = 0; s < m; ++s)
            for (int a = 0; a < D_W; ++a)
                for (int b = 0; b < D_W; ++b)
                    F4d[t_idx][u][s](a, b) = flat(u * D_W + a, s * D_W + b);
}

// ============================================================
// compute_filter_kernels
// ============================================================
void compute_filter_kernels(const Kernel2D& X, const Mat3& Pi, int obs_index,
                            const std::array<double, N>& obs_gain,
                            int filter_iters, double relax,
                            Kernel2D& R, Eigen::MatrixXd& Q_flat,
                            Kernel3D& F, Kernel2D& Xhat) {
    const int dw = D_W;

    // Zero initialize
    for (auto& row : R)
        for (auto& v : row)
            v.setZero();
    for (int i = 0; i < N; ++i)
        for (int j2 = 0; j2 < N; ++j2)
            for (int k = 0; k < N; ++k)
                F[i][j2][k].setZero();
    for (auto& row : Xhat)
        for (auto& v : row)
            v.setZero();

    int max_flat = N * dw;
    Q_flat.setZero(max_flat, max_flat);

    // F_flat_store[j] is a (N*dw) x (N*dw) matrix (only first (j+1)*dw used)
    std::vector<Eigen::MatrixXd> F_flat_store(N,
        Eigen::MatrixXd::Zero(max_flat, max_flat));

    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;

    for (int j = 0; j < N; ++j) {
        double g = obs_gain[j];
        double prec = g * g;
        int m = j + 1;
        int mdw = m * dw;

        // Compute base = X[j,:m] - X[j,:m] @ Pi  and Q_corr
        // R[j] = base - Q_corr - Gamma
        // base[s] = X[j][s] - Pi * X[j][s]  (note: Pi acts on rows)
        // Actually: base = Xj - Xj @ Pi → base[s] = X[j][s] - X[j][s].transpose() * Pi
        // Wait: X[j][s] is a Vec3. "Xj @ Pi" means matrix multiply where Xj is (m, dw)
        // and Pi is (dw, dw). So row s: X[j][s] @ Pi = (Pi^T * X[j][s]) but since Pi
        // is diagonal, Pi^T = Pi, so it's just Pi * X[j][s].
        // Actually in numpy: Xj @ Pi means each row of Xj is multiplied by Pi on the right.
        // For row vector x: x @ Pi = (Pi^T x)^T. For a column vector stored as Vec3:
        // the equivalent is (Pi^T * x) = Pi * x (since Pi is symmetric diagonal).

        // Q_corr[s] = dt * sum over rows of (Xj.ravel() @ Q_flat[:mdw, :mdw])
        // reshaped to (m, dw). This is Xj_flat^T @ Q_flat reshaped.
        Eigen::VectorXd Xj_flat(mdw);
        for (int s = 0; s < m; ++s)
            Xj_flat.segment<D_W>(s * dw) = X[j][s];

        Eigen::VectorXd Q_corr_flat = DT * (Xj_flat.transpose() *
            Q_flat.topLeftCorner(mdw, mdw)).transpose();

        // Gamma computation
        std::vector<Vec3> Gamma(m, Vec3::Zero());

        if (j > 0) {
            // x_obs = X[j][:j, obs_index], gains_s = obs_gain[:j]
            // coeff_a = x_obs * gains_s
            // R_past = R[:j, :m] → R[z][s] for z in [0,j), s in [0,m)
            // weights[z][s] = coeff_a[z] * R[z][s]
            // total_sum[s] = sum_z weights[z][s]
            // cum[z][s] = cumsum over z of weights
            // inclusive[s] = cum[min(s,j-1)][s] if s < j, else total_sum[s]
            // gamma_a[s] = dt * (total_sum[s] - inclusive[s])

            std::vector<double> coeff_a(j);
            for (int z = 0; z < j; ++z)
                coeff_a[z] = X[j][z](obs_index) * obs_gain[z];

            // total_sum and inclusive
            std::vector<Vec3> total_sum(m, Vec3::Zero());
            // cumulative sum storage: cum[z][s]
            std::vector<std::vector<Vec3>> cum(j, std::vector<Vec3>(m, Vec3::Zero()));

            for (int z = 0; z < j; ++z) {
                for (int s = 0; s < m; ++s) {
                    Vec3 w = coeff_a[z] * R[z][s];
                    total_sum[s] += w;
                    if (z == 0)
                        cum[z][s] = w;
                    else
                        cum[z][s] = cum[z - 1][s] + w;
                }
            }

            int jj = std::min(j, m);
            std::vector<Vec3> inclusive(m, Vec3::Zero());
            for (int s = 0; s < jj; ++s)
                inclusive[s] = cum[s][s]; // cum[s][s] for s < j
            for (int s = j; s < m; ++s)
                inclusive[s] = total_sum[s];

            // gamma_a
            std::vector<Vec3> gamma_a(m);
            for (int s = 0; s < m; ++s)
                gamma_a[s] = DT * (total_sum[s] - inclusive[s]);

            // gamma_b: dot_mat[z][s] = X[j][z] . R[z][s] (dot product)
            // cum_dot[z][s] = cumsum over z of dot_mat
            // b_scalar[s] = ... (see Python)
            // gamma_b[s] = dt * b_scalar[s] * obs_gain[s] * e_i

            std::vector<std::vector<double>> dot_mat(j, std::vector<double>(m, 0.0));
            for (int z = 0; z < j; ++z)
                for (int s = 0; s < m; ++s)
                    dot_mat[z][s] = X[j][z].dot(R[z][s]);

            std::vector<std::vector<double>> cum_dot(j, std::vector<double>(m, 0.0));
            for (int z = 0; z < j; ++z)
                for (int s = 0; s < m; ++s) {
                    cum_dot[z][s] = dot_mat[z][s];
                    if (z > 0) cum_dot[z][s] += cum_dot[z - 1][s];
                }

            std::vector<double> b_scalar(m, 0.0);
            for (int s = 1; s < jj; ++s)
                b_scalar[s] = cum_dot[s - 1][s];
            if (m > j && j > 0)
                for (int s = j; s < m; ++s)
                    b_scalar[s] = cum_dot[j - 1][s];

            for (int s = 0; s < m; ++s) {
                Vec3 gamma_b_s = (DT * b_scalar[s] * obs_gain[s]) * e_i;
                Gamma[s] = gamma_a[s] + gamma_b_s;
            }
        }

        // R[j][s] = base[s] - Q_corr[s] - Gamma[s]
        for (int s = 0; s < m; ++s) {
            Vec3 base_s = X[j][s] - Pi * X[j][s];
            Vec3 Q_corr_s = Q_corr_flat.segment<D_W>(s * dw);
            R[j][s] = base_s - Q_corr_s - Gamma[s];
        }

        // Xhat computation
        if (j == 0) {
            Xhat[j][0] = Pi * X[j][0];
        } else {
            int jdw = j * dw;
            Eigen::MatrixXd& F_base = F_flat_store[j];
            F_base.setZero(max_flat, max_flat);

            // Copy from previous
            F_base.topLeftCorner(jdw, jdw) =
                F_flat_store[j - 1].topLeftCorner(jdw, jdw);

            // Rj_j_flat = R[j][:j].ravel()
            Eigen::VectorXd Rj_j_flat(jdw);
            for (int s = 0; s < j; ++s)
                Rj_j_flat.segment<D_W>(s * dw) = R[j][s];

            // F_base[:jdw, jdw:mdw] = g * outer(Rj_j_flat, e_i)
            Eigen::MatrixXd outer1 = g * Rj_j_flat * e_i.transpose();
            F_base.block(0, jdw, jdw, dw) = outer1;
            F_base.block(jdw, 0, dw, jdw) = outer1.transpose();

            Eigen::MatrixXd F_cur = F_base.topLeftCorner(mdw, mdw);

            // Fixed-point iteration
            double dt_prec = DT * prec;
            double one_minus_relax = 1.0 - relax;

            // XjPi[s] = Pi * X[j][s]
            Eigen::VectorXd XjPi_flat(mdw);
            for (int s = 0; s < m; ++s)
                XjPi_flat.segment<D_W>(s * dw) = Pi * X[j][s];

            for (int iter = 0; iter < filter_iters; ++iter) {
                // Xhat_j = XjPi + dt * (Xj_flat^T @ F_cur)^T reshaped
                Eigen::VectorXd Xhat_j_flat =
                    XjPi_flat + DT * (Xj_flat.head(mdw).transpose() * F_cur).transpose();

                // diff_flat = (X[j][:j] - Xhat_j[:j]).ravel()
                Eigen::VectorXd diff_flat(jdw);
                for (int s = 0; s < j; ++s) {
                    Vec3 xhat_s(Xhat_j_flat.segment<D_W>(s * dw));
                    diff_flat.segment<D_W>(s * dw) = X[j][s] - xhat_s;
                }

                Eigen::MatrixXd F_new_inner =
                    F_base.topLeftCorner(jdw, jdw) +
                    dt_prec * Rj_j_flat * diff_flat.transpose();

                F_cur.topLeftCorner(jdw, jdw) =
                    relax * F_new_inner +
                    one_minus_relax * F_cur.topLeftCorner(jdw, jdw);
            }

            F_flat_store[j].topLeftCorner(mdw, mdw) = F_cur;

            // Final Xhat
            Eigen::VectorXd Xhat_j_flat =
                XjPi_flat + DT * (Xj_flat.head(mdw).transpose() * F_cur).transpose();
            for (int s = 0; s < m; ++s)
                Xhat[j][s] = Xhat_j_flat.segment<D_W>(s * dw);
        }

        // Update Q_flat
        if (j < N - 1) {
            Eigen::VectorXd Rj_flat(mdw);
            for (int s = 0; s < m; ++s)
                Rj_flat.segment<D_W>(s * dw) = R[j][s];
            Q_flat.topLeftCorner(mdw, mdw) +=
                DT * prec * Rj_flat * Rj_flat.transpose();
        }
    }

    // Convert F_flat_store to Kernel3D F
    for (int j = 0; j < N; ++j) {
        int m = j + 1;
        int mdw = m * dw;
        if (mdw > 0) {
            Eigen::MatrixXd flat_j = F_flat_store[j].topLeftCorner(mdw, mdw);
            from_flat(flat_j, m, F, j);
        }
    }
}

// ============================================================
// primitive_control_kernel
// ============================================================
void primitive_control_kernel(const Kernel2D& D, const Kernel3D& F,
                              const Mat3& Pi, Kernel2D& calD) {
    for (auto& row : calD)
        for (auto& v : row)
            v.setZero();

    for (int j = 0; j < N; ++j) {
        int m = j + 1;
        int mdw = m * D_W;

        // calD[j][s] = D[j][:m] @ Pi + dt * (D[j].ravel() @ F_t).reshape
        // i.e., calD[j][s] = Pi * D[j][s]  +  dt * sum_u F[j][u][s]^T * D[j][u]
        // (using the to_flat convention)

        // More precisely: compute via flat multiply
        Eigen::VectorXd Dj_flat(mdw);
        for (int s = 0; s < m; ++s)
            Dj_flat.segment<D_W>(s * D_W) = D[j][s];

        Eigen::MatrixXd F_t;
        to_flat(F, j, m, F_t);

        Eigen::VectorXd result_flat(mdw);
        // Pi * D[j][s] for each s
        for (int s = 0; s < m; ++s)
            result_flat.segment<D_W>(s * D_W) = Pi * D[j][s];

        // + dt * (Dj_flat^T @ F_t)^T
        result_flat += DT * (Dj_flat.transpose() * F_t).transpose();

        for (int s = 0; s < m; ++s)
            calD[j][s] = result_flat.segment<D_W>(s * D_W);
    }
}

// ============================================================
// forward_environment
// ============================================================
EnvironmentResult forward_environment(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {

    EnvironmentResult env;

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
    Eigen::MatrixXd Q1, Q2;
    Kernel3D F1, F2;

    for (int it = 0; it < inner_iters; ++it) {
        compute_filter_kernels(X, Pi_1, obs_idx_1, gain1,
                               FILTER_INNER_ITERS, FILTER_RELAX,
                               R1, Q1, F1, Xhat1);
        compute_filter_kernels(X, Pi_2, obs_idx_2, gain2,
                               FILTER_INNER_ITERS, FILTER_RELAX,
                               R2, Q2, F2, Xhat2);

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
    env.Q1 = Q1; env.Q2 = Q2;
    env.F1 = F1; env.F2 = F2;
    env.Xhat1 = Xhat1; env.Xhat2 = Xhat2;
    env.calD1 = calD1; env.calD2 = calD2;
    return env;
}

// ============================================================
// backward_kernels
// ============================================================
void backward_kernels(const Kernel2D& X, const Kernel2D& Rk,
                      const Kernel2D& Dk,
                      const std::array<double, N>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel3D& Hk) {
    // Zero init
    for (auto& row : Hx)
        for (auto& v : row)
            v.setZero();
    for (int i = 0; i < N; ++i)
        for (int j2 = 0; j2 < N; ++j2)
            for (int k = 0; k < N; ++k)
                Hk[i][j2][k].setZero();

    // Terminal condition
    for (int s = 0; s < N; ++s)
        Hx[N - 1][s] = -terminal_state_weight * X[N - 1][s];

    for (int j = N - 2; j >= 0; --j) {
        int t_idx = j + 1;
        double Pk = prec_k[t_idx];
        int m = j + 1;
        int mz = t_idx + 1;

        // coupling_x = dt * Pk * einsum("za,zrab->rb", Rz, Hk_t)
        // Rz = Rk[t_idx][:mz], Hk_t = Hk[t_idx][:mz][:m]
        // coupling_x[r][b] = dt * Pk * sum_z sum_a Rk[t_idx][z][a] * Hk[t_idx][z][r](a,b)

        // Hx[j][r] = Hx[t_idx][r] + dt * (X[t_idx][r] + coupling_x_r)
        for (int r = 0; r < m; ++r) {
            Vec3 coupling = Vec3::Zero();
            for (int z = 0; z < mz; ++z) {
                // sum_a Rk[t_idx][z][a] * Hk[t_idx][z][r](a, :)
                // = Hk[t_idx][z][r]^T * Rk[t_idx][z]
                coupling += Hk[t_idx][z][r].transpose() * Rk[t_idx][z];
            }
            coupling *= DT * Pk;
            Hx[j][r] = Hx[t_idx][r] + DT * (X[t_idx][r] + coupling);
        }

        // Hk[j][z][r](a,b) = Hk[t_idx][z][r](a,b) + dt * (first - coupling_k)
        // first = D[t_idx][r][a] * Hx[t_idx][z][b]  → outer product D[t_idx][r] * Hx[t_idx][z]^T
        // Actually: first[z,r,a,b] = Dk[t_idx,r,a,_] * Hx[t_idx,_,z,b]
        // Python: first = Dk[t_idx, :m, :, None] * Hx[t_idx, None, :m, None, :]
        // Dk[t_idx][:m] shape (m, dw) → Dk[t_idx][r][a]
        // Hx[t_idx][:m] shape (m, dw) → Hx[t_idx][z][b]
        // first[r,z,a,b] = Dk[t_idx][r][a] * Hx[t_idx][z][b]

        // W[r][b] = dt * Pk * sum_z sum_c Rk[t_idx][z][c] * Hk[t_idx][z][r](c,b)
        // = coupling from above (same as coupling_x but different usage)
        // Actually W is same formula: W[r,b] = dt*Pk * einsum("zc,zrcb->rb", Rz, Hk_t)
        // This is the same as coupling_x above.

        // coupling_k[z,r,a,b] = X[t_idx][z][a] * W[r][b]
        // Python: X[t_idx, :m, None, :, None] * W[None, :, None, :]
        // coupling_k[z,r,a,b] = X[t_idx][z][a] * W[r][b]

        // First compute W for each r
        std::vector<Vec3> W(m, Vec3::Zero());
        for (int r = 0; r < m; ++r) {
            for (int z = 0; z < mz; ++z)
                W[r] += Hk[t_idx][z][r].transpose() * Rk[t_idx][z];
            W[r] *= DT * Pk;
        }

        for (int z = 0; z < m; ++z) {
            for (int r = 0; r < m; ++r) {
                // first = Dk[t_idx][r] * Hx[t_idx][z]^T  (outer product, 3x3)
                Mat3 first = Dk[t_idx][r] * Hx[t_idx][z].transpose();
                // coupling_k = X[t_idx][z] * W[r]^T
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

        // I_val = dt * Pk * dot(Rk[t_idx][:mz].ravel(), barHk[t_idx][:mz].ravel())
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

    for (int it = 1; it <= MAX_PICARD_ITERS; ++it) {
        auto env = forward_environment(D1, D2, p1_val, p2_val,
                                       FORWARD_INNER_ITERS,
                                       Pi_1, obs_idx_1, Pi_2, obs_idx_2);

        Kernel2D Hx1, Hx2;
        Kernel3D Hk1, Hk2;
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

    auto env = forward_environment(D1, D2, p1_val, p2_val,
                                   FORWARD_INNER_ITERS,
                                   Pi_1, obs_idx_1, Pi_2, obs_idx_2);

    return {D1, D2, env, residuals};
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
