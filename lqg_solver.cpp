// Decentralized LQG noise-state game solver — F-free implementation
//
// The 36MB Kernel3D F is never materialized during solving. Products
// like sum_u F[j][u][s]^T * v[u] are computed from the rank-1
// decomposition F[j][u][s] = border + sum_k Xtilde[k][u] * A_store[k][s]^T.

#include "lqg_solver.h"
#include <algorithm>

// Runtime grid parameters
int g_n = 40;
double g_T = 1.0;
double g_dt = 1.0 / 39.0;

void set_grid(int n, double T) {
    g_n = std::max(4, std::min(n, N_MAX));
    g_T = T;
    g_dt = g_T / (g_n - 1);
}

double g_b1 = B1_DEFAULT;
double g_b2 = B2_DEFAULT;
double g_r1 = RHO;
double g_r2 = RHO;
double g_sigma = 1.0;

// --- state_kernel_from_calD ---

void state_kernel_from_calD(const Kernel2D& calD1, const Kernel2D& calD2,
                            Kernel2D& X) {
    X.setZero();
    Vec3 sigE0 = g_sigma * E0();
    for (int s = 0; s < g_n; ++s) {
        X[s][s] = sigE0;
        Vec3 cumsum = Vec3::Zero();
        for (int t = s + 1; t < g_n; ++t) {
            cumsum += g_dt * (calD1[t - 1][s] + calD2[t - 1][s]);
            X[t][s] = sigE0 + cumsum;
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
    double dt2_prec = g_dt * g_dt * prec;

    for (int j = 0; j < g_n; ++j) {
        if (j == 0) {
            Xtilde[0][0] = I_minus_Pi * X[0][0];
            A_store[0][0].setZero();
            continue;
        }

        // c_k = dot(Xtilde[k][0..k], X[j][0..k]),  partial_c_k = same but excluding u=k
        std::array<double, N_MAX> c_k, partial_c_k;
        for (int k = 0; k < j; ++k) {
            double acc = 0.0;
            for (int u = 0; u < k; ++u)
                acc += Xtilde[k][u].dot(X[j][u]);
            partial_c_k[k] = acc;
            c_k[k] = acc + Xtilde[k][k].dot(X[j][k]);
        }

        // coeff_a[z] = g * X[j][z](obs_index)  (for gamma_a accumulation)
        std::array<double, N_MAX> coeff_a;
        for (int z = 0; z < j; ++z)
            coeff_a[z] = g * X[j][z](obs_index);

        // Per-s: accumulate correction in one pass over k
        std::array<Vec3, N_MAX> correction;
        for (int s = 0; s < j; ++s) {
            Vec3 q_upper = Vec3::Zero();
            Vec3 ga_acc = Vec3::Zero();

            for (int k = s + 1; k < j; ++k) {
                Vec3 Xtks = Xtilde[k][s];
                q_upper += c_k[k] * Xtks;
                ga_acc += coeff_a[k] * Xtks;
            }

            Vec3 gamma_a_s = g_dt * ga_acc;
            correction[s] = dt2_prec * (c_k[s] * Xtilde[s][s] + q_upper) + gamma_a_s;
        }

        // Xtilde = (I-Pi)*X - correction
        for (int s = 0; s < j; ++s)
            Xtilde[j][s] = I_minus_Pi * X[j][s] - correction[s];
        Xtilde[j][j] = I_minus_Pi * X[j][j];

        // --- Closed-form Kalman gain (replaces iterative relaxation) ---
        double sigma_minus = 0.0;
        for (int u = 0; u < j; ++u)
            sigma_minus += g_dt * Xtilde[j][u].squaredNorm();

        double scale = 1.0 / (1.0 + g_dt * prec * sigma_minus);
        for (int s = 0; s <= j; ++s)
            Xtilde[j][s] *= scale;

        double dt_prec = g_dt * prec;
        for (int s = 0; s < j; ++s)
            A_store[j][s] = dt_prec * Xtilde[j][s];
    }
}

// --- primitive_control_kernel (F-free) ---
//
// calD[j][s] = Pi*D[j][s] + g_dt * sum_u F[j][u][s]^T * D[j][u]
// decomposed into border_gt, border_lt, and interior sums.

void primitive_control_kernel(
    const Kernel2D& D, const Kernel2D& Xtilde, const Kernel2D& A_store,
    double obs_gain_val, int obs_index, const Mat3& Pi, Kernel2D& calD) {

    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain_val;
    double DT_g = g_dt * g;

    for (int j = 0; j < g_n; ++j) {
        if (j == 0) {
            calD[0][0] = Pi * D[0][0];
            continue;
        }

        // partial_d_k = sum_{u<k} dot(Xtilde[k][u], D[j][u])
        std::array<double, N_MAX> partial_d_k;
        partial_d_k[0] = 0.0;
        for (int k = 1; k <= j; ++k) {
            double acc = 0.0;
            for (int u = 0; u < k; ++u)
                acc += Xtilde[k][u].dot(D[j][u]);
            partial_d_k[k] = acc;
        }

        // D_obs[u] = D[j][u](obs_index) for border_gt
        std::array<double, N_MAX> D_obs;
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
            acc += DT_g * bg + g_dt * ia;
            acc += DT_g * partial_d_k[s] * e_i;  // border_lt

            calD[j][s] = acc;
        }

        // s = j: only border_lt contributes
        calD[j][j] = Pi * D[j][j] + DT_g * partial_d_k[j] * e_i;
    }
}

// --- CE-based filter: incremental rank-1 projection ---
//
// Builds M_j incrementally for j = 0..n-1 via rank-1 updates.
// At each j, extracts Xtilde[j] = (I - M_j) X[j] and calD[j] = M_j D[j].

// Thread-local buffers for compute_ce_filter_and_calD.
// Avoids repeated heap allocation of the V matrix (~600KB at N=160)
// and scratch vectors across Picard iterations.
struct CEFilterBufs {
    int dim_alloc = 0;
    Eigen::MatrixXd V;
    Eigen::VectorXd h, v, Xvec, Dvec, coeff;
    void ensure(int dim, int n_obs) {
        if (dim_alloc == dim) return;
        dim_alloc = dim;
        V.resize(dim, n_obs);
        h.resize(dim); v.resize(dim); Xvec.resize(dim); Dvec.resize(dim);
        coeff.resize(n_obs);
    }
};
static thread_local CEFilterBufs s_ce_bufs;

static void compute_ce_filter_and_calD(
    const Kernel2D& X, const Kernel2D& D,
    int obs_idx, double obs_gain,
    Kernel2D& Xtilde, Kernel2D& calD) {

    int n = g_n;
    int dim = 3 * n;
    int n_obs = n - 1;
    double g = obs_gain;

    // Thin orthonormal basis: M_j = V_j V_j^T where V_j has j columns.
    // Mat-vec M*x = V*(V^T*x) costs O(active*j) instead of O(dim*j).
    //
    // Key: at step j, all vectors (h, Xvec, Dvec) and all V columns are
    // zero below row 3(j+1). So mat-vecs are restricted to the top
    // "active" = 3(j+1) rows, cutting average cost by ~2x.
    s_ce_bufs.ensure(dim, n_obs);
    auto& V = s_ce_bufs.V;
    auto& h = s_ce_bufs.h;
    auto& v = s_ce_bufs.v;
    auto& Xvec = s_ce_bufs.Xvec;
    auto& Dvec = s_ce_bufs.Dvec;
    auto& coeff = s_ce_bufs.coeff;

    V.setZero();
    int rank = 0;
    h.setZero();
    v.setZero();
    Xvec.setZero();
    Dvec.setZero();

    // j = 0: no observations yet, M_0 = 0
    Xtilde[0][0] = X[0][0];
    calD[0][0].setZero();

    for (int j = 1; j < n; ++j) {
        int active = 3 * (j + 1);  // nonzero rows at step j
        auto Vact = V.topRows(active).leftCols(rank);

        // Build h_j (only active entries)
        h.head(active).setZero();
        for (int z = 0; z <= j; ++z)
            for (int k = 0; k < 3; ++k)
                h(3*z + k) = g * g_dt * X[j][z](k);
        h(3*j + obs_idx) += 1.0;

        // Innovation: v = (I - V V^T) h, restricted to active rows
        v.head(active) = h.head(active);
        if (rank > 0) {
            auto c = coeff.head(rank);
            c.noalias() = Vact.transpose() * h.head(active);
            v.head(active).noalias() -= Vact * c;
        }
        double vnorm = v.head(active).norm();
        if (vnorm > 1e-15) {
            V.col(rank).head(active) = v.head(active) / vnorm;
            rank++;
        }

        // Refresh Vact after potential rank increase
        auto Vr = V.topRows(active).leftCols(rank);
        auto c = coeff.head(rank);

        // Xtilde[j] = (I - V V^T) X_vec_j
        for (int z = 0; z <= j; ++z)
            Xvec.segment<3>(3*z) = X[j][z];
        if (rank > 0) {
            c.noalias() = Vr.transpose() * Xvec.head(active);
            Xvec.head(active).noalias() -= Vr * c;
        }
        for (int z = 0; z <= j; ++z)
            Xtilde[j][z] = Xvec.segment<3>(3*z);

        // calD[j] = V V^T D_vec_j
        for (int z = 0; z <= j; ++z)
            Dvec.segment<3>(3*z) = D[j][z];
        if (rank > 0) {
            c.noalias() = Vr.transpose() * Dvec.head(active);
            Dvec.head(active).noalias() = Vr * c;
        } else {
            Dvec.head(active).setZero();
        }
        for (int z = 0; z <= j; ++z)
            calD[j][z] = Dvec.segment<3>(3*z);
    }
}

// Forward environment using discrete CE projection.
// Replaces compute_filter_kernels + primitive_control_kernel with
// the exact discrete conditional expectation at each time step.

static void forward_environment_ce(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env) {

    // Initial calD from Pi*D (same seed as standard forward_environment)
    // Temporaries declared once outside the inner loop to avoid
    // repeated Kernel2D construction (~300KB each at N=160).
    Kernel2D calD1, calD2, X, calD1_new, calD2_new, X_new;
    for (int j = 0; j < g_n; ++j)
        for (int s = 0; s <= j; ++s) {
            calD1[j][s] = Pi_1 * D1[j][s];
            calD2[j][s] = Pi_2 * D2[j][s];
        }

    state_kernel_from_calD(calD1, calD2, X);

    for (int it = 0; it < inner_iters; ++it) {
        #pragma omp parallel sections
        {
            #pragma omp section
            compute_ce_filter_and_calD(X, D1, obs_idx_1, obs_gain1,
                                        env.Xtilde1, calD1_new);
            #pragma omp section
            compute_ce_filter_and_calD(X, D2, obs_idx_2, obs_gain2,
                                        env.Xtilde2, calD2_new);
        }

        state_kernel_from_calD(calD1_new, calD2_new, X_new);
        for (int t = 0; t < g_n; ++t)
            for (int s = 0; s <= t; ++s)
                X[t][s] = 0.6 * X_new[t][s] + 0.4 * X[t][s];
    }

    env.X = X;
    env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
    env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
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
    for (int j = 0; j < g_n; ++j)
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
        for (int t = 0; t < g_n; ++t)
            for (int s = 0; s <= t; ++s)
                X[t][s] = 0.6 * X_new[t][s] + 0.4 * X[t][s];
    }

    env.X = X;
    env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
    env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
}

// --- backward_kernels (ping-pong, no Kernel3D) ---
//
// Two N_MAX×N_MAX Mat3 slices replace the full Kernel3D.

// Dynamically sized HkSlice: g_n × g_n Mat3 entries.
struct HkSlice {
    int n;
    std::vector<Mat3> data;
    HkSlice() : n(0) {}
    void resize(int nn) {
        if (n != nn) { n = nn; data.resize(nn * nn); }
    }
    Mat3& operator()(int z, int r) { return data[z * n + r]; }
    const Mat3& operator()(int z, int r) const { return data[z * n + r]; }
};

// Thread-local HkSlice buffers — reused across backward_kernels calls within a
// thread to avoid heap allocation churn.
// thread_local is needed because OMP parallel sections call backward_kernels
// concurrently from different threads.
static thread_local std::unique_ptr<HkSlice> s_hk_buf0, s_hk_buf1;

static void ensure_hk_buffers() {
    if (!s_hk_buf0) s_hk_buf0 = std::make_unique<HkSlice>();
    if (!s_hk_buf1) s_hk_buf1 = std::make_unique<HkSlice>();
    s_hk_buf0->resize(g_n);
    s_hk_buf1->resize(g_n);
}

void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N_MAX>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx) {
    Hx.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = -terminal_state_weight * X[g_n - 1][s];

    ensure_hk_buffers();
    // Only zero the portion we'll use (g_n rows × g_n cols)
    for (int z = 0; z < g_n; ++z)
        for (int r = 0; r < g_n; ++r)
            (*s_hk_buf0)(z, r).setZero();
    HkSlice* cur = s_hk_buf0.get();
    HkSlice* nxt = s_hk_buf1.get();

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;  // t+1
        double Pk = prec_k[tp];

        // W[r] = g_dt * Pk * sum_z Hk[tp][z][r]^T * Xtilde[tp][z]
        std::array<Vec3, N_MAX> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += (*cur)(z, r).transpose() * Xtildek[tp][z];
            W[r] = g_dt * Pk * acc;
        }

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx[tp][r] + g_dt * (X[tp][r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                (*nxt)(z, r) = (*cur)(z, r)
                    + g_dt * (Dk[tp][r] * Hx[tp][z].transpose()
                          - X[tp][z] * W[r].transpose());

        std::swap(cur, nxt);
    }
}

// Version that also outputs the kernel information wedge V^i(t,r).
// V[t][r] = g_dt * Pk * sum_z Hk[t][z][r]^T * Xtilde[t][z]
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N_MAX>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel2D& Vkernel) {
    Hx.setZero();
    Vkernel.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = -terminal_state_weight * X[g_n - 1][s];

    ensure_hk_buffers();
    for (int z = 0; z < g_n; ++z)
        for (int r = 0; r < g_n; ++r)
            (*s_hk_buf0)(z, r).setZero();
    HkSlice* cur = s_hk_buf0.get();
    HkSlice* nxt = s_hk_buf1.get();

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];

        std::array<Vec3, N_MAX> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += (*cur)(z, r).transpose() * Xtildek[tp][z];
            W[r] = g_dt * Pk * acc;
        }

        // Store W[r] = V^i(t_{j+1}, r) into Vkernel
        for (int r = 0; r <= j; ++r)
            Vkernel[tp][r] = W[r];

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx[tp][r] + g_dt * (X[tp][r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                (*nxt)(z, r) = (*cur)(z, r)
                    + g_dt * (Dk[tp][r] * Hx[tp][z].transpose()
                          - X[tp][z] * W[r].transpose());

        std::swap(cur, nxt);
    }
}

// Legacy version that also fills Hk (for figure output)
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N_MAX>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel3D& Hk) {
    Hx.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = -terminal_state_weight * X[g_n - 1][s];

    for (int z = 0; z < g_n; ++z)
        for (int r = 0; r < g_n; ++r)
            Hk[g_n - 1][z][r].setZero();

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];

        std::array<Vec3, N_MAX> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += Hk[tp][z][r].transpose() * Xtildek[tp][z];
            W[r] = g_dt * Pk * acc;
        }

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx[tp][r] + g_dt * (X[tp][r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                Hk[j][z][r] = Hk[tp][z][r]
                    + g_dt * (Dk[tp][r] * Hx[tp][z].transpose()
                          - X[tp][z] * W[r].transpose());
    }
}

// --- backward_bar_adjoints ---

BackwardBarResult backward_bar_adjoints(
    const Kernel2D& X, const Kernel2D& Xtildek, const Kernel2D& Dk,
    const std::array<double, N_MAX>& barX, double b,
    const std::array<double, N_MAX>& prec_k, double terminal_weight) {

    BackwardBarResult res;
    res.barHx.fill(0.0);
    res.barHk.setZero();
    res.barHx[g_n - 1] = terminal_weight * (barX[g_n - 1] - b);

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];

        double I_val = 0.0;
        for (int z = 0; z <= tp; ++z)
            I_val += Xtildek[tp][z].dot(res.barHk[tp][z]);
        I_val *= g_dt * Pk;

        res.barHx[j] = res.barHx[tp] + g_dt * ((barX[tp] - b) + I_val);

        for (int s = 0; s <= j; ++s)
            res.barHk[j][s] = res.barHk[tp][s]
                + g_dt * (Dk[tp][s] * res.barHx[tp] - X[tp][s] * I_val);
    }
    return res;
}

// --- solve_bar_equilibrium ---

BarSolution solve_bar_equilibrium(
    const EnvironmentResult& env, const Kernel2D& D1, const Kernel2D& D2,
    double prec1, double prec2,
    int max_iters, double relax, double tol, bool /*verbose*/) {

    std::array<double, N_MAX> barD1{}, barD2{};
    auto prec1_arr = make_constant_prec(prec1);
    auto prec2_arr = make_constant_prec(prec2);
    double last_err = 1e30;

    for (int it = 1; it <= max_iters; ++it) {
        std::array<double, N_MAX> barX;
        barX[0] = X0;
        for (int j = 0; j < g_n - 1; ++j)
            barX[j + 1] = barX[j] + g_dt * (barD1[j] + barD2[j]);

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

        std::array<double, N_MAX> barD1_new, barD2_new;
        for (int j = 0; j < g_n; ++j) {
            barD1_new[j] = -(1.0 / g_r1) * bba1.barHx[j];
            barD2_new[j] = -(1.0 / g_r2) * bba2.barHx[j];
        }

        double n1 = 0, d1 = 0, n2 = 0, d2 = 0;
        for (int j = 0; j < g_n; ++j) {
            double diff1 = barD1_new[j] - barD1[j];
            double diff2 = barD2_new[j] - barD2[j];
            d1 += diff1 * diff1;  n1 += barD1_new[j] * barD1_new[j];
            d2 += diff2 * diff2;  n2 += barD2_new[j] * barD2_new[j];
        }
        last_err = std::max(std::sqrt(d1) / std::max(1.0, std::sqrt(n1)),
                            std::sqrt(d2) / std::max(1.0, std::sqrt(n2)));

        for (int j = 0; j < g_n; ++j) {
            barD1[j] += relax * (barD1_new[j] - barD1[j]);
            barD2[j] += relax * (barD2_new[j] - barD2[j]);
        }
        if (last_err < tol) break;
    }

    BarSolution sol;
    sol.barX[0] = X0;
    for (int j = 0; j < g_n - 1; ++j)
        sol.barX[j + 1] = sol.barX[j] + g_dt * (barD1[j] + barD2[j]);
    sol.barD1 = barD1;
    sol.barD2 = barD2;
    sol.bar_residual = last_err;
    return sol;
}

// --- kernel_dot: SIMD-optimized dot over packed triangular data ---

static double kernel_dot(const Kernel2D& a1, const Kernel2D& a2,
                         const Kernel2D& b1, const Kernel2D& b2) {
    const int FLAT = g_n * (g_n + 1) / 2 * 3;
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

// Core solver: takes initial D1, D2 (may be zero or warm-started)
static EquilibriumResult solve_equilibrium_core(
    double p1_val, double p2_val,
    Kernel2D D1, Kernel2D D2, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {

    std::vector<double> residuals;
    auto prec1 = make_constant_prec(p1_val * p1_val);
    auto prec2 = make_constant_prec(p2_val * p2_val);
    EnvironmentResult env;

    // Anderson acceleration (AA) state
    constexpr int AA_M = 5;
    constexpr int AA_STORE = AA_M + 1;  // +1 so write slot doesn't collide with reads
    constexpr double AA_REG = 1e-12;
    constexpr double AA_BETA = 0.6;
    const int TRI = g_n * (g_n + 1) / 2;

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

        // Compute G = -(1/r_k)*Hx and residual F = G - D
        const double neg_inv_r1 = -(1.0 / g_r1);
        const double neg_inv_r2 = -(1.0 / g_r2);
        auto& cur = hist[stored % AA_STORE];
        double norm_f = 0.0, norm_g = 0.0;
        for (int i = 0; i < TRI; ++i) {
            cur.g1.data[i] = neg_inv_r1 * Hx1.data[i];
            cur.g2.data[i] = neg_inv_r2 * Hx2.data[i];
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
                    double fifj = kernel_dot(hist[ii].f1, hist[ii].f2, hist[jj].f1, hist[jj].f2);
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

EquilibriumResult solve_equilibrium(
    double p1_val, double p2_val, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {
    Kernel2D D1, D2;
    D1.setZero();
    D2.setZero();
    return solve_equilibrium_core(p1_val, p2_val, std::move(D1), std::move(D2),
                                  verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2);
}

EquilibriumResult solve_equilibrium_warm(
    double p1_val, double p2_val,
    const Kernel2D& D1_init, const Kernel2D& D2_init,
    bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {
    Kernel2D D1(D1_init), D2(D2_init);
    return solve_equilibrium_core(p1_val, p2_val, std::move(D1), std::move(D2),
                                  verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2);
}

// --- solve_equilibrium_ce: Picard iteration with discrete CE projection ---

EquilibriumResult solve_equilibrium_ce(
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

    // Anderson acceleration state (same as standard solver)
    constexpr int AA_M = 5;
    constexpr int AA_STORE = AA_M + 1;
    constexpr double AA_REG = 1e-12;
    constexpr double AA_BETA = 0.6;
    const int TRI = g_n * (g_n + 1) / 2;

    struct AAEntry { Kernel2D f1, f2, g1, g2; };
    std::vector<AAEntry> hist(AA_STORE);
    int stored = 0;

    for (int it = 1; it <= MAX_PICARD_ITERS; ++it) {
        // Forward pass using discrete CE
        forward_environment_ce(D1, D2, p1_val, p2_val, FORWARD_INNER_ITERS,
                               Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);

        // Backward pass (uses Xtilde, same as standard solver)
        Kernel2D Hx1, Hx2;
        #pragma omp parallel sections
        {
            #pragma omp section
            backward_kernels(env.X, env.Xtilde2, D2, prec2, 0.0, Hx1);
            #pragma omp section
            backward_kernels(env.X, env.Xtilde1, D1, prec1, 0.0, Hx2);
        }

        // Residual and Anderson acceleration (identical to standard solver)
        const double neg_inv_r1 = -(1.0 / g_r1);
        const double neg_inv_r2 = -(1.0 / g_r2);
        auto& cur = hist[stored % AA_STORE];
        double norm_f = 0.0, norm_g = 0.0;
        for (int i = 0; i < TRI; ++i) {
            cur.g1.data[i] = neg_inv_r1 * Hx1.data[i];
            cur.g2.data[i] = neg_inv_r2 * Hx2.data[i];
            cur.f1.data[i] = cur.g1.data[i] - D1.data[i];
            cur.f2.data[i] = cur.g2.data[i] - D2.data[i];
            norm_f += cur.f1.data[i].squaredNorm() + cur.f2.data[i].squaredNorm();
            norm_g += cur.g1.data[i].squaredNorm() + cur.g2.data[i].squaredNorm();
        }

        double err = std::sqrt(norm_f) / std::max(1.0, std::sqrt(norm_g));
        residuals.push_back(err);
        if (!std::isfinite(err)) break;

        if (verbose && (it <= 5 || it % 50 == 0))
            std::cout << "  [CE] it=" << it << "  resid=" << err << std::endl;
        if (err < PICARD_TOL) {
            if (verbose)
                std::cout << "  [CE] Converged at iteration " << it << std::endl;
            break;
        }

        int m = std::min(stored, AA_M);
        if (m >= 1) {
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
                    double fifj = kernel_dot(hist[ii].f1, hist[ii].f2, hist[jj].f1, hist[jj].f2);
                    H(i, j) = H(j, i) = norm_f - Ffi(i) - Ffi(j) + fifj;
                }
            }
            for (int i = 0; i < m; ++i)
                H(i, i) += AA_REG * (1.0 + H(i, i));

            Eigen::VectorXd gamma = H.ldlt().solve(rhs);

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
            for (int i = 0; i < TRI; ++i) {
                D1.data[i] += PICARD_RELAX * cur.f1.data[i];
                D2.data[i] += PICARD_RELAX * cur.f2.data[i];
            }
        }
        stored++;
    }

    // After convergence: run standard filter once to populate A_store
    // (needed for F materialization and API compatibility)
    {
        Kernel2D xt_tmp1, xt_tmp2;
        compute_filter_kernels(env.X, Pi_1, obs_idx_1, p1_val,
                               FILTER_INNER_ITERS, FILTER_RELAX,
                               xt_tmp1, env.A_store1);
        compute_filter_kernels(env.X, Pi_2, obs_idx_2, p2_val,
                               FILTER_INNER_ITERS, FILTER_RELAX,
                               xt_tmp2, env.A_store2);
    }

    // Compute calD using CE projection on converged X and D
    Kernel2D calD1, calD2;
    #pragma omp parallel sections
    {
        #pragma omp section
        compute_ce_filter_and_calD(env.X, D1, obs_idx_1, p1_val,
                                    env.Xtilde1, calD1);
        #pragma omp section
        compute_ce_filter_and_calD(env.X, D2, obs_idx_2, p2_val,
                                    env.Xtilde2, calD2);
    }

    return {D1, D2, std::move(env), std::move(calD1), std::move(calD2), residuals};
}

// --- compute_costs_general ---

CostPair compute_costs_general(const EnvironmentResult& env,
                               const Kernel2D& calD1, const Kernel2D& calD2,
                               const BarSolution& bar_sol,
                               double r1_val, double r2_val,
                               double b1_val, double b2_val) {
    double J1 = 0.0, J2 = 0.0;
    for (int j = 0; j < g_n; ++j) {
        double var_X = 0.0, var_D1 = 0.0, var_D2 = 0.0;
        for (int s = 0; s < j; ++s) {
            var_X += g_dt * env.X[j][s].squaredNorm();
            var_D1 += g_dt * calD1[j][s].squaredNorm();
            var_D2 += g_dt * calD2[j][s].squaredNorm();
        }
        double dx1 = bar_sol.barX[j] - b1_val;
        double dx2 = bar_sol.barX[j] - b2_val;
        J1 += g_dt * (dx1*dx1 + var_X + r1_val * (bar_sol.barD1[j]*bar_sol.barD1[j] + var_D1));
        J2 += g_dt * (dx2*dx2 + var_X + r2_val * (bar_sol.barD2[j]*bar_sol.barD2[j] + var_D2));
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
    for (int j = 1; j < g_n; ++j) {
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

// --- compute_F_slice_at_T (memory-efficient) ---
//
// Computes F[g_n-1][u][s] using ping-pong of two N_MAX×N_MAX slices
// instead of the full 3D kernel. Memory: ~2 * N_MAX^2 * 72 bytes ≈ 0.9MB.

std::unique_ptr<FSlice> compute_F_slice_at(const Kernel2D& Xtilde, const Kernel2D& A_store,
                                            double obs_gain, int obs_index, int t_idx) {
    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain;

    // Two slices for ping-pong: prev = F[j-1], cur = F[j]
    // Heap-allocated; pointer swap avoids deep copy.
    auto prev = std::make_unique<FSlice>();
    auto cur  = std::make_unique<FSlice>();

    // j = 0: F[0][0][0] = 0
    (*prev)(0, 0).setZero();

    for (int j = 1; j <= t_idx; ++j) {
        // Borders
        for (int u = 0; u < j; ++u) {
            (*cur)(u, j) = g * Xtilde[j][u] * e_i.transpose();
            (*cur)(j, u) = g * e_i * Xtilde[j][u].transpose();
        }
        (*cur)(j, j).setZero();

        // Interior: F[j][u][s] = F[j-1][u][s] + Xtilde[j][u] * A_store[j][s]^T
        for (int u = 0; u < j; ++u)
            for (int s = 0; s < j; ++s)
                (*cur)(u, s) = (*prev)(u, s) + Xtilde[j][u] * A_store[j][s].transpose();

        std::swap(prev, cur);
    }

    // After the loop, prev holds F[t_idx]
    return prev;
}

std::unique_ptr<FSlice> compute_F_slice_at_T(const Kernel2D& Xtilde, const Kernel2D& A_store,
                                              double obs_gain, int obs_index) {
    return compute_F_slice_at(Xtilde, A_store, obs_gain, obs_index, g_n - 1);
}

// --- Exact discrete conditional expectation ---

DiscreteProjection discrete_conditional_expectation(
    const Kernel2D& X, int obs_idx, double obs_gain, int t_idx) {

    int n = t_idx + 1;
    int dim = 3 * n;
    int n_obs = t_idx;  // observations at j = 1, ..., t_idx
    double g = obs_gain;

    // 1. Build measurement matrix H: n_obs × dim
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_obs, dim);
    for (int j = 1; j <= t_idx; ++j) {
        int row = j - 1;
        for (int z = 0; z <= j; ++z)
            for (int k = 0; k < 3; ++k)
                H(row, 3*z + k) = g * g_dt * X[j][z](k);
        H(row, 3*j + obs_idx) += 1.0;
    }

    // 2. Factorize HH^T (n_obs × n_obs, much smaller than dim × dim)
    Eigen::MatrixXd HHt = H * H.transpose();
    auto ldlt = HHt.ldlt();

    // 3. Extract Xtilde without forming the full dim × dim M matrix.
    //    (I - M) Xmat = Xmat - H^T (HH^T)^{-1} (H Xmat)
    //    Peak memory: H (n_obs × dim) + HXmat (n_obs × n) instead of M (dim × dim).
    Kernel2D Xt_out;
    Eigen::MatrixXd Xmat = Eigen::MatrixXd::Zero(dim, n);
    for (int j = 0; j <= t_idx; ++j)
        for (int z = 0; z <= j; ++z)
            Xmat.block<3,1>(3*z, j) = X[j][z];

    Eigen::MatrixXd HXmat = H * Xmat;              // n_obs × n
    Eigen::MatrixXd solved = ldlt.solve(HXmat);     // n_obs × n
    Eigen::MatrixXd Xt_mat = Xmat - H.transpose() * solved;  // dim × n

    for (int j = 0; j <= t_idx; ++j)
        for (int z = 0; z <= j; ++z)
            Xt_out[j][z] = Xt_mat.block<3,1>(3*z, j);

    // 4. Diagnostics from the LDLT factorization.
    //    Rank = number of positive pivots.
    //    Idempotency: compute trace(M) = trace(H^T (HH^T)^{-1} H)
    //    = trace((HH^T)^{-1} HH^T) = trace(I_{n_obs}) = n_obs for full rank.
    //    Check via ||H (HH^T)^{-1} H^T H - H||_F / ||H||_F.
    int rank = 0;
    for (int i = 0; i < n_obs; ++i)
        if (ldlt.vectorD()(i) > 1e-12) rank++;

    // Cheap idempotency check: ||(HH^T)^{-1}(HH^T) - I||_F
    Eigen::MatrixXd I_check = ldlt.solve(HHt);
    double idem = (I_check - Eigen::MatrixXd::Identity(n_obs, n_obs)).norm()
                  / std::max(1e-15, std::sqrt(static_cast<double>(n_obs)));

    return {Eigen::MatrixXd(), std::move(Xt_out), rank, idem};
}

IncrementalProjection build_projections_incremental(
    const Kernel2D& X, int obs_idx, double obs_gain) {

    int n = g_n;
    int dim = 3 * n;
    int n_obs = n - 1;
    double g = obs_gain;

    // Thin V basis instead of dense M (saves dim×dim = 1.8MB at N=160)
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(dim, n_obs);
    int rank = 0;
    Eigen::VectorXd h(dim), v(dim), coeff(n_obs);
    h.setZero();
    v.setZero();

    for (int j = 1; j < n; ++j) {
        int active = 3 * (j + 1);
        h.head(active).setZero();
        for (int z = 0; z <= j; ++z)
            for (int k = 0; k < 3; ++k)
                h(3*z + k) = g * g_dt * X[j][z](k);
        h(3*j + obs_idx) += 1.0;

        v.head(active) = h.head(active);
        if (rank > 0) {
            auto Vact = V.topRows(active).leftCols(rank);
            auto c = coeff.head(rank);
            c.noalias() = Vact.transpose() * h.head(active);
            v.head(active).noalias() -= Vact * c;
        }
        double vnorm = v.head(active).norm();
        if (vnorm > 1e-15) {
            V.col(rank).head(active) = v.head(active) / vnorm;
            rank++;
        }
    }

    // Extract Xtilde using V: (I - VV^T) Xmat
    Kernel2D Xt_out;
    Eigen::MatrixXd Xmat = Eigen::MatrixXd::Zero(dim, n);
    for (int j = 0; j < n; ++j)
        for (int z = 0; z <= j; ++z)
            Xmat.block<3,1>(3*z, j) = X[j][z];

    // VtX = V^T * Xmat (rank × n), then Xt = Xmat - V * VtX
    Eigen::MatrixXd VtX = V.leftCols(rank).transpose() * Xmat;
    Eigen::MatrixXd Xt_mat = Xmat - V.leftCols(rank) * VtX;

    for (int j = 0; j < n; ++j)
        for (int z = 0; z <= j; ++z)
            Xt_out[j][z] = Xt_mat.block<3,1>(3*z, j);

    // Idempotency: VV^T is exact projection by construction, check V^T V = I
    Eigen::MatrixXd VtV = V.leftCols(rank).transpose() * V.leftCols(rank);
    double idem = (VtV - Eigen::MatrixXd::Identity(rank, rank)).norm()
                  / std::max(1e-15, std::sqrt(static_cast<double>(rank)));

    return {Eigen::MatrixXd(), std::move(Xt_out), idem};
}

ProjectionPair exact_discrete_CE(const EquilibriumResult& eq) {
    // Reconstruct X from converged primitive control kernels
    Kernel2D X;
    state_kernel_from_calD(eq.calD1, eq.calD2, X);

    int t_idx = g_n - 1;
    DiscreteProjection p1, p2;
    #pragma omp parallel sections
    {
        #pragma omp section
        p1 = discrete_conditional_expectation(X, eq.env.obs_idx1, eq.env.obs_gain1, t_idx);
        #pragma omp section
        p2 = discrete_conditional_expectation(X, eq.env.obs_idx2, eq.env.obs_gain2, t_idx);
    }
    return {std::move(p1), std::move(p2)};
}
