// Decentralized LQG noise-state game solver — F-free implementation
//
// The 36MB Kernel3D F is never materialized during solving. Products
// like sum_u F[j][u][s]^T * v[u] are computed from the rank-1
// decomposition F[j][u][s] = border + sum_k Xtilde[k][u] * A_store[k][s]^T.

#include "lqg_solver.h"
#include <algorithm>

double g_b1 = B1_DEFAULT;
double g_b2 = B2_DEFAULT;

// --- state_kernel_from_calD ---

void state_kernel_from_calD(const Kernel2D& calD1, const Kernel2D& calD2,
                            Kernel2D& X) {
    X.setZero();
    for (int s = 0; s < N; ++s) {
        X[s][s] = E0();
        Vec3 cumsum = Vec3::Zero();
        for (int t = s + 1; t < N; ++t) {
            cumsum += DT * (calD1[t - 1][s] + calD2[t - 1][s]);
            X[t][s] = E0() + cumsum;
        }
    }
}

// --- compute_filter_kernels (F-free) ---
//
// Computes Xtilde and A_store without materializing F.
// gamma_b (border_lt) is identically zero by causality:
// Xtilde[u][s] = 0 for s > u.

static void compute_filter_kernels(
    const Kernel2D& X, const Mat3& Pi, int obs_index,
    double obs_gain_val, int filter_iters, double relax,
    Kernel2D& Xtilde, Kernel2D& A_store) {

    Mat3 I_minus_Pi = Mat3::Identity() - Pi;
    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;

    double g = obs_gain_val;
    double prec = g * g;
    double dt2_prec = DT * DT * prec;

    for (int j = 0; j < N; ++j) {
        if (j == 0) {
            Xtilde[0][0] = I_minus_Pi * X[0][0];
            A_store[0][0].setZero();
            continue;
        }

        // c_k = dot(Xtilde[k][0..k], X[j][0..k]),  partial_c_k = same but excluding u=k
        std::array<double, N> c_k, partial_c_k;
        for (int k = 0; k < j; ++k) {
            double acc = 0.0;
            for (int u = 0; u < k; ++u)
                acc += Xtilde[k][u].dot(X[j][u]);
            partial_c_k[k] = acc;
            c_k[k] = acc + Xtilde[k][k].dot(X[j][k]);
        }

        // coeff_a[z] = g * X[j][z](obs_index)  (for gamma_a accumulation)
        std::array<double, N> coeff_a;
        for (int z = 0; z < j; ++z)
            coeff_a[z] = g * X[j][z](obs_index);

        // Per-s: accumulate Q_corr, gamma_a, and Xhat_const in one pass over k
        std::array<Vec3, N> correction, Xhat_const;
        for (int s = 0; s < j; ++s) {
            Vec3 q_upper = Vec3::Zero();
            Vec3 ga_acc = Vec3::Zero();
            Vec3 xi_acc = Vec3::Zero();

            for (int k = s + 1; k < j; ++k) {
                Vec3 Xtks = Xtilde[k][s];
                q_upper += c_k[k] * Xtks;
                ga_acc += coeff_a[k] * Xtks;
                xi_acc += partial_c_k[k] * A_store[k][s];
            }

            Vec3 gamma_a_s = DT * ga_acc;
            correction[s] = dt2_prec * (c_k[s] * Xtilde[s][s] + q_upper) + gamma_a_s;

            Vec3 xhat = Pi * X[j][s] + gamma_a_s + DT * xi_acc;
            xhat += DT * g * partial_c_k[s] * e_i;
            Xhat_const[s] = xhat;
        }

        // Xtilde = (I-Pi)*X - correction
        for (int s = 0; s < j; ++s)
            Xtilde[j][s] = I_minus_Pi * X[j][s] - correction[s];
        Xtilde[j][j] = I_minus_Pi * X[j][j];

        // Rank-1 filter iteration for A_store[j]
        double sigma = 0.0;
        for (int u = 0; u < j; ++u)
            sigma += DT * Xtilde[j][u].dot(X[j][u]);

        double alpha = relax * DT * prec;
        double beta = 1.0 - relax;

        std::array<Vec3, N> A;
        for (int s = 0; s < j; ++s)
            A[s].setZero();

        for (int iter = 0; iter < filter_iters; ++iter)
            for (int s = 0; s < j; ++s) {
                Vec3 diff = X[j][s] - Xhat_const[s] - sigma * A[s];
                A[s] = alpha * diff + beta * A[s];
            }

        for (int s = 0; s < j; ++s)
            A_store[j][s] = A[s];
    }
}

// --- primitive_control_kernel (F-free) ---
//
// calD[j][s] = Pi*D[j][s] + DT * sum_u F[j][u][s]^T * D[j][u]
// decomposed into border_gt, border_lt, and interior sums.

void primitive_control_kernel(
    const Kernel2D& D, const Kernel2D& Xtilde, const Kernel2D& A_store,
    double obs_gain_val, int obs_index, const Mat3& Pi, Kernel2D& calD) {

    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain_val;
    double DT_g = DT * g;

    for (int j = 0; j < N; ++j) {
        if (j == 0) {
            calD[0][0] = Pi * D[0][0];
            continue;
        }

        // partial_d_k = sum_{u<k} dot(Xtilde[k][u], D[j][u])
        std::array<double, N> partial_d_k;
        partial_d_k[0] = 0.0;
        for (int k = 1; k <= j; ++k) {
            double acc = 0.0;
            for (int u = 0; u < k; ++u)
                acc += Xtilde[k][u].dot(D[j][u]);
            partial_d_k[k] = acc;
        }

        // D_obs[u] = D[j][u](obs_index) for border_gt
        std::array<double, N> D_obs;
        for (int u = 0; u <= j; ++u)
            D_obs[u] = D[j][u](obs_index);

        for (int s = 0; s < j; ++s) {
            Vec3 acc = Pi * D[j][s];

            // border_gt + interior merged over k = s+1..j
            Vec3 bg = Vec3::Zero(), ia = Vec3::Zero();
            for (int k = s + 1; k <= j; ++k) {
                bg += D_obs[k] * Xtilde[k][s];
                ia += partial_d_k[k] * A_store[k][s];
            }
            acc += DT_g * bg + DT * ia;
            acc += DT_g * partial_d_k[s] * e_i;  // border_lt

            calD[j][s] = acc;
        }

        // s = j: only border_lt contributes
        calD[j][j] = Pi * D[j][j] + DT_g * partial_d_k[j] * e_i;
    }
}

// --- forward_environment ---

void forward_environment(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env) {

    Kernel2D calD1, calD2;
    for (int j = 0; j < N; ++j)
        for (int s = 0; s <= j; ++s) {
            calD1[j][s] = Pi_1 * D1[j][s];
            calD2[j][s] = Pi_2 * D2[j][s];
        }

    Kernel2D X;
    state_kernel_from_calD(calD1, calD2, X);

    Kernel2D calD1_new, calD2_new, X_new;
    for (int it = 0; it < inner_iters; ++it) {
        #pragma omp parallel sections
        {
            #pragma omp section
            compute_filter_kernels(X, Pi_1, obs_idx_1, obs_gain1,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   env.Xtilde1, env.A_store1);
            #pragma omp section
            compute_filter_kernels(X, Pi_2, obs_idx_2, obs_gain2,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   env.Xtilde2, env.A_store2);
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            primitive_control_kernel(D1, env.Xtilde1, env.A_store1,
                                     obs_gain1, obs_idx_1, Pi_1, calD1_new);
            #pragma omp section
            primitive_control_kernel(D2, env.Xtilde2, env.A_store2,
                                     obs_gain2, obs_idx_2, Pi_2, calD2_new);
        }

        state_kernel_from_calD(calD1_new, calD2_new, X_new);
        for (int t = 0; t < N; ++t)
            for (int s = 0; s <= t; ++s)
                X[t][s] = 0.6 * X_new[t][s] + 0.4 * X[t][s];
    }

    env.X = X;
    env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
    env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
}

// --- backward_kernels (ping-pong, no Kernel3D) ---
//
// Two N×N Mat3 slices (~230KB total) replace the 36MB Kernel3D.

using HkSlice = std::array<std::array<Mat3, N>, N>;

void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx) {
    Hx.setZero();
    for (int s = 0; s < N; ++s)
        Hx[N - 1][s] = -terminal_state_weight * X[N - 1][s];

    HkSlice buf0{}, buf1;
    for (auto& row : buf0) for (auto& m : row) m.setZero();
    HkSlice* cur = &buf0;
    HkSlice* nxt = &buf1;

    for (int j = N - 2; j >= 0; --j) {
        int tp = j + 1;  // t+1
        double Pk = prec_k[tp];

        // W[r] = DT * Pk * sum_z Hk[tp][z][r]^T * Xtilde[tp][z]
        std::array<Vec3, N> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += (*cur)[z][r].transpose() * Xtildek[tp][z];
            W[r] = DT * Pk * acc;
        }

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx[tp][r] + DT * (X[tp][r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                (*nxt)[z][r] = (*cur)[z][r]
                    + DT * (Dk[tp][r] * Hx[tp][z].transpose()
                          - X[tp][z] * W[r].transpose());

        std::swap(cur, nxt);
    }
}

// Legacy version that also fills Hk (for figure output)
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel3D& Hk) {
    Hx.setZero();
    for (int s = 0; s < N; ++s)
        Hx[N - 1][s] = -terminal_state_weight * X[N - 1][s];

    for (int z = 0; z < N; ++z)
        for (int r = 0; r < N; ++r)
            Hk[N - 1][z][r].setZero();

    for (int j = N - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];

        std::array<Vec3, N> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += Hk[tp][z][r].transpose() * Xtildek[tp][z];
            W[r] = DT * Pk * acc;
        }

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx[tp][r] + DT * (X[tp][r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                Hk[j][z][r] = Hk[tp][z][r]
                    + DT * (Dk[tp][r] * Hx[tp][z].transpose()
                          - X[tp][z] * W[r].transpose());
    }
}

// --- backward_bar_adjoints ---

BackwardBarResult backward_bar_adjoints(
    const Kernel2D& X, const Kernel2D& Xtildek, const Kernel2D& Dk,
    const std::array<double, N>& barX, double b,
    const std::array<double, N>& prec_k, double terminal_weight) {

    BackwardBarResult res;
    res.barHx.fill(0.0);
    res.barHk.setZero();
    res.barHx[N - 1] = terminal_weight * (barX[N - 1] - b);

    for (int j = N - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];

        double I_val = 0.0;
        for (int z = 0; z <= tp; ++z)
            I_val += Xtildek[tp][z].dot(res.barHk[tp][z]);
        I_val *= DT * Pk;

        res.barHx[j] = res.barHx[tp] + DT * ((barX[tp] - b) + I_val);

        for (int s = 0; s <= j; ++s)
            res.barHk[j][s] = res.barHk[tp][s]
                + DT * (Dk[tp][s] * res.barHx[tp] - X[tp][s] * I_val);
    }
    return res;
}

// --- solve_bar_equilibrium ---

BarSolution solve_bar_equilibrium(
    const EnvironmentResult& env, const Kernel2D& D1, const Kernel2D& D2,
    double prec1, double prec2,
    int max_iters, double relax, double tol, bool /*verbose*/) {

    std::array<double, N> barD1{}, barD2{};
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
            bba1 = backward_bar_adjoints(env.X, env.Xtilde2, D2,
                                          barX, g_b1, prec2_arr, 0.0);
            #pragma omp section
            bba2 = backward_bar_adjoints(env.X, env.Xtilde1, D1,
                                          barX, g_b2, prec1_arr, 0.0);
        }

        std::array<double, N> barD1_new, barD2_new;
        for (int j = 0; j < N; ++j) {
            barD1_new[j] = -(1.0 / RHO) * bba1.barHx[j];
            barD2_new[j] = -(1.0 / RHO) * bba2.barHx[j];
        }

        double n1 = 0, d1 = 0, n2 = 0, d2 = 0;
        for (int j = 0; j < N; ++j) {
            double diff1 = barD1_new[j] - barD1[j];
            double diff2 = barD2_new[j] - barD2[j];
            d1 += diff1 * diff1;  n1 += barD1_new[j] * barD1_new[j];
            d2 += diff2 * diff2;  n2 += barD2_new[j] * barD2_new[j];
        }
        last_err = std::max(std::sqrt(d1) / std::max(1.0, std::sqrt(n1)),
                            std::sqrt(d2) / std::max(1.0, std::sqrt(n2)));

        for (int j = 0; j < N; ++j) {
            barD1[j] += relax * (barD1_new[j] - barD1[j]);
            barD2[j] += relax * (barD2_new[j] - barD2[j]);
        }
        if (last_err < tol) break;
    }

    BarSolution sol;
    sol.barX[0] = X0;
    for (int j = 0; j < N - 1; ++j)
        sol.barX[j + 1] = sol.barX[j] + DT * (barD1[j] + barD2[j]);
    sol.barD1 = barD1;
    sol.barD2 = barD2;
    sol.bar_residual = last_err;
    return sol;
}

// --- kernel_dot: SIMD-optimized dot over packed triangular data ---

static double kernel_dot(const Kernel2D& a1, const Kernel2D& a2,
                         const Kernel2D& b1, const Kernel2D& b2) {
    constexpr int FLAT = Kernel2D::TRI_SIZE * 3;
    Eigen::Map<const Eigen::VectorXd> va1(a1.data[0].data(), FLAT);
    Eigen::Map<const Eigen::VectorXd> vb1(b1.data[0].data(), FLAT);
    Eigen::Map<const Eigen::VectorXd> va2(a2.data[0].data(), FLAT);
    Eigen::Map<const Eigen::VectorXd> vb2(b2.data[0].data(), FLAT);
    return va1.dot(vb1) + va2.dot(vb2);
}

// --- solve_equilibrium: Anderson-accelerated Picard iteration ---
//
// Anderson(m=5) uses the last m residuals to find optimal mixing
// coefficients, typically reducing iteration count by 2-3x vs Picard.

EquilibriumResult solve_equilibrium(
    double p1_val, double p2_val, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {

    Kernel2D D1, D2;
    D1.setZero();
    D2.setZero();

    std::vector<double> residuals;
    auto prec1 = make_constant_prec(p1_val * p1_val);
    auto prec2 = make_constant_prec(p2_val * p2_val);
    EnvironmentResult env;

    // Anderson acceleration (AA) state
    constexpr int AA_M = 5;
    constexpr int AA_STORE = AA_M + 1;  // +1 so write slot doesn't collide with reads
    constexpr double AA_REG = 1e-12;
    constexpr double AA_BETA = 0.6;
    constexpr int TRI = Kernel2D::TRI_SIZE;

    struct AAEntry { Kernel2D f1, f2, g1, g2; };
    std::vector<AAEntry> hist(AA_STORE);
    int stored = 0;

    for (int it = 1; it <= MAX_PICARD_ITERS; ++it) {
        forward_environment(D1, D2, p1_val, p2_val, FORWARD_INNER_ITERS,
                            Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);

        Kernel2D Hx1, Hx2;
        #pragma omp parallel sections
        {
            #pragma omp section
            backward_kernels(env.X, env.Xtilde2, D2, prec2, 0.0, Hx1);
            #pragma omp section
            backward_kernels(env.X, env.Xtilde1, D1, prec1, 0.0, Hx2);
        }

        // Compute G = -(1/rho)*Hx and residual F = G - D
        constexpr double neg_inv_rho = -(1.0 / RHO);
        auto& cur = hist[stored % AA_STORE];
        double norm_f = 0.0, norm_g = 0.0;
        for (int i = 0; i < TRI; ++i) {
            cur.g1.data[i] = neg_inv_rho * Hx1.data[i];
            cur.g2.data[i] = neg_inv_rho * Hx2.data[i];
            cur.f1.data[i] = cur.g1.data[i] - D1.data[i];
            cur.f2.data[i] = cur.g2.data[i] - D2.data[i];
            norm_f += cur.f1.data[i].squaredNorm() + cur.f2.data[i].squaredNorm();
            norm_g += cur.g1.data[i].squaredNorm() + cur.g2.data[i].squaredNorm();
        }

        double err = std::sqrt(norm_f) / std::max(1.0, std::sqrt(norm_g));
        residuals.push_back(err);
        if (!std::isfinite(err)) break;

        if (verbose && (it <= 5 || it % 50 == 0))
            std::cout << "  it=" << it << "  resid=" << err << std::endl;
        if (err < PICARD_TOL) {
            if (verbose)
                std::cout << "  Converged at iteration " << it << std::endl;
            break;
        }

        // --- Anderson acceleration ---
        int m = std::min(stored, AA_M);

        if (m >= 1) {
            // Build Gram matrix: H[i][j] = <dF_i, dF_j> where dF_i = F_curr - f_hist[i]
            // Expanded as <F,F> - <F,f_j> - <f_i,F> + <f_i,f_j>
            Eigen::VectorXd Ffi(m);
            for (int i = 0; i < m; ++i) {
                int idx = (stored - 1 - i + AA_STORE) % AA_STORE;
                Ffi(i) = kernel_dot(cur.f1, cur.f2, hist[idx].f1, hist[idx].f2);
            }

            Eigen::MatrixXd H(m, m);
            Eigen::VectorXd rhs(m);
            for (int i = 0; i < m; ++i) {
                rhs(i) = norm_f - Ffi(i);
                for (int j = 0; j <= i; ++j) {
                    int ii = (stored - 1 - i + AA_STORE) % AA_STORE;
                    int jj = (stored - 1 - j + AA_STORE) % AA_STORE;
                    double fifj = kernel_dot(hist[ii].f1, hist[ii].f2,
                                             hist[jj].f1, hist[jj].f2);
                    H(i, j) = H(j, i) = norm_f - Ffi(i) - Ffi(j) + fifj;
                }
            }
            for (int i = 0; i < m; ++i)
                H(i, i) += AA_REG * (1.0 + H(i, i));

            Eigen::VectorXd gamma = H.ldlt().solve(rhs);

            // Mix: D_aa = G - sum_i gamma_i * (G - g_hist[i]),  D = (1-beta)*D + beta*D_aa
            std::array<const Vec3*, AA_M> g1p, g2p;
            std::array<double, AA_M> gam;
            for (int i = 0; i < m; ++i) {
                int idx = (stored - 1 - i + AA_STORE) % AA_STORE;
                g1p[i] = hist[idx].g1.data.data();
                g2p[i] = hist[idx].g2.data.data();
                gam[i] = gamma(i);
            }
            const Vec3* cg1 = cur.g1.data.data();
            const Vec3* cg2 = cur.g2.data.data();
            for (int i = 0; i < TRI; ++i) {
                Vec3 d1_aa = cg1[i], d2_aa = cg2[i];
                for (int k = 0; k < m; ++k) {
                    d1_aa -= gam[k] * (cg1[i] - g1p[k][i]);
                    d2_aa -= gam[k] * (cg2[i] - g2p[k][i]);
                }
                D1.data[i] = (1.0 - AA_BETA) * D1.data[i] + AA_BETA * d1_aa;
                D2.data[i] = (1.0 - AA_BETA) * D2.data[i] + AA_BETA * d2_aa;
            }
        } else {
            // First iteration: simple Picard relaxation
            for (int i = 0; i < TRI; ++i) {
                D1.data[i] += PICARD_RELAX * cur.f1.data[i];
                D2.data[i] += PICARD_RELAX * cur.f2.data[i];
            }
        }
        stored++;
    }

    // Convergence check breaks before Anderson update, so env matches D1/D2
    Kernel2D calD1, calD2;
    #pragma omp parallel sections
    {
        #pragma omp section
        primitive_control_kernel(D1, env.Xtilde1, env.A_store1,
                                 p1_val, obs_idx_1, Pi_1, calD1);
        #pragma omp section
        primitive_control_kernel(D2, env.Xtilde2, env.A_store2,
                                 p2_val, obs_idx_2, Pi_2, calD2);
    }

    return {D1, D2, std::move(env), std::move(calD1), std::move(calD2), residuals};
}

// --- compute_costs_general ---

CostPair compute_costs_general(const EnvironmentResult& env,
                               const Kernel2D& calD1, const Kernel2D& calD2,
                               const BarSolution& bar_sol,
                               double r_val, double b1_val, double b2_val) {
    double J1 = 0.0, J2 = 0.0;
    for (int j = 0; j < N; ++j) {
        double var_X = 0.0, var_D1 = 0.0, var_D2 = 0.0;
        for (int s = 0; s < j; ++s) {
            var_X += DT * env.X[j][s].squaredNorm();
            var_D1 += DT * calD1[j][s].squaredNorm();
            var_D2 += DT * calD2[j][s].squaredNorm();
        }
        double dx1 = bar_sol.barX[j] - b1_val;
        double dx2 = bar_sol.barX[j] - b2_val;
        J1 += DT * (dx1*dx1 + var_X + r_val * (bar_sol.barD1[j]*bar_sol.barD1[j] + var_D1));
        J2 += DT * (dx2*dx2 + var_X + r_val * (bar_sol.barD2[j]*bar_sol.barD2[j] + var_D2));
    }
    return {J1, J2};
}

// --- materialize_F (for figure output only) ---
//
// Builds F[j][u][s] incrementally from Xtilde + A_store.
// F[j][u][s] = F[j-1][u][s] + Xtilde[j][u] * A_store[j][s]^T  (interior)
// Borders: F[j][u][j] = g*Xtilde[j][u]*e_i^T, F[j][j][s] = g*e_i*Xtilde[j][s]^T

void materialize_F(const Kernel2D& Xtilde, const Kernel2D& A_store,
                   double obs_gain, int obs_index, Kernel3D& F) {
    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain;

    F[0][0][0].setZero();
    for (int j = 1; j < N; ++j) {
        for (int u = 0; u < j; ++u) {
            F[j][u][j] = g * Xtilde[j][u] * e_i.transpose();
            F[j][j][u] = g * e_i * Xtilde[j][u].transpose();
        }
        F[j][j][j].setZero();

        for (int u = 0; u < j; ++u)
            for (int s = 0; s < j; ++s)
                F[j][u][s] = F[j - 1][u][s] + Xtilde[j][u] * A_store[j][s].transpose();
    }
}
