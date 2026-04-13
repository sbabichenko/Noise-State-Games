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
double g_x0 = 0.0;

SolverContext SolverContext::capture_current() {
    return SolverContext{g_n, g_T, g_b1, g_b2, g_r1, g_r2, g_sigma, g_x0};
}

void SolverContext::apply() const {
    set_grid(n, T);
    g_b1 = b1;
    g_b2 = b2;
    g_r1 = r1;
    g_r2 = r2;
    g_sigma = sigma;
    g_x0 = x0;
}

ScopedSolverContext::ScopedSolverContext(const SolverContext& next)
    : previous_(SolverContext::capture_current()) {
    next.apply();
}

ScopedSolverContext::~ScopedSolverContext() {
    previous_.apply();
}

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
    Kernel2D& Xtilde, Kernel2D* A_store = nullptr,
    std::array<double, N_MAX>* scale_out = nullptr) {

    Mat3 I_minus_Pi = Mat3::Identity() - Pi;
    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;

    double g = obs_gain_val;
    double prec = g * g;
    double dt2_prec = g_dt * g_dt * prec;

    for (int j = 0; j < g_n; ++j) {
        if (j == 0) {
            Xtilde[0][0] = I_minus_Pi * X[0][0];
            if (A_store) (*A_store)[0][0].setZero();
            if (scale_out) (*scale_out)[0] = 1.0;
            continue;
        }

        // Pre-extract row pointer to avoid repeated triangular index math
        const Vec3* X_j = &X[j][0];

        // c_k = dot(Xtilde[k][0..k], X[j][0..k]),  partial_c_k = same but excluding u=k
        std::array<double, N_MAX> c_k, partial_c_k;
        for (int k = 0; k < j; ++k) {
            double acc = 0.0;
            const Vec3* Xt_k = &Xtilde[k][0];
            for (int u = 0; u < k; ++u)
                acc += Xt_k[u].dot(X_j[u]);
            partial_c_k[k] = acc;
            c_k[k] = acc + Xt_k[k].dot(X_j[k]);
        }

        // coeff_a[z] = g * X[j][z](obs_index)  (for gamma_a accumulation)
        std::array<double, N_MAX> coeff_a;
        for (int z = 0; z < j; ++z)
            coeff_a[z] = g * X_j[z](obs_index);

        // Compute Xtilde[j][s] = (I-Pi)*X[j][s] - correction[s] in one fused pass
        for (int s = 0; s < j; ++s) {
            Vec3 q_upper = Vec3::Zero();
            Vec3 ga_acc = Vec3::Zero();

            for (int k = s + 1; k < j; ++k) {
                Vec3 Xtks = Xtilde[k][s];
                q_upper += c_k[k] * Xtks;
                ga_acc += coeff_a[k] * Xtks;
            }

            Xtilde[j][s] = I_minus_Pi * X_j[s]
                - dt2_prec * (c_k[s] * Xtilde[s][s] + q_upper)
                - g_dt * ga_acc;
        }
        Xtilde[j][j] = I_minus_Pi * X_j[j];

        // --- Closed-form Kalman gain (replaces iterative relaxation) ---
        double sigma_minus = 0.0;
        for (int u = 0; u < j; ++u)
            sigma_minus += g_dt * Xtilde[j][u].squaredNorm();

        double scale = 1.0 / (1.0 + g_dt * prec * sigma_minus);
        for (int s = 0; s <= j; ++s)
            Xtilde[j][s] *= scale;

        if (A_store) {
            double dt_prec = g_dt * prec;
            for (int s = 0; s < j; ++s)
                (*A_store)[j][s] = dt_prec * Xtilde[j][s];
        }
        if (scale_out) (*scale_out)[j] = scale;
    }
}

// --- primitive_control_kernel (F-free) ---
//
// calD[j][s] = Pi*D[j][s] + g_dt * sum_u F[j][u][s]^T * D[j][u]
// decomposed into border_gt, border_lt, and interior sums.

void primitive_control_kernel(
    const Kernel2D& D, const Kernel2D& Xtilde,
    double obs_gain_val, int obs_index, const Mat3& Pi, Kernel2D& calD) {

    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain_val;
    double DT_g = g_dt * g;
    // A_store[k][s] = (g_dt*g²)*Xtilde[k][s]; original used g_dt*sum(partial_d*A_store)
    // = g_dt²*g² * sum(partial_d*Xtilde). So coefficient for ia is g_dt²*g².
    double dt2_prec = g_dt * g_dt * g * g;

    for (int j = 0; j < g_n; ++j) {
        if (j == 0) {
            calD[0][0] = Pi * D[0][0];
            continue;
        }

        // Pre-extract row pointer to avoid repeated triangular index math
        const Vec3* D_j = &D[j][0];

        // partial_d_k = sum_{u<k} dot(Xtilde[k][u], D[j][u])
        std::array<double, N_MAX> partial_d_k;
        partial_d_k[0] = 0.0;
        for (int k = 1; k <= j; ++k) {
            double acc = 0.0;
            const Vec3* Xt_k = &Xtilde[k][0];
            for (int u = 0; u < k; ++u)
                acc += Xt_k[u].dot(D_j[u]);
            partial_d_k[k] = acc;
        }

        // D_obs[u] = D[j][u](obs_index) for border_gt
        std::array<double, N_MAX> D_obs;
        for (int u = 0; u <= j; ++u)
            D_obs[u] = D_j[u](obs_index);

        for (int s = 0; s < j; ++s) {
            Vec3 acc = Pi * D_j[s];

            // border_gt + interior merged over k = s+1..j
            Vec3 bg = Vec3::Zero(), ia = Vec3::Zero();
            for (int k = s + 1; k <= j; ++k) {
                Vec3 Xtks = Xtilde[k][s];
                bg += D_obs[k] * Xtks;
                ia += partial_d_k[k] * Xtks;  // A_store = dt_prec * Xtilde
            }
            acc += DT_g * bg + dt2_prec * ia;
            acc += DT_g * partial_d_k[s] * e_i;  // border_lt

            calD[j][s] = acc;
        }

        // s = j: only border_lt contributes
        calD[j][j] = Pi * D_j[j] + DT_g * partial_d_k[j] * e_i;
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

// Thread-local reusable buffers for forward_environment_ce.
struct FwdEnvCEBufs {
    Kernel2D calD1, calD2, calD1_new, calD2_new, X_new;
    int n_alloc = 0;
    void ensure(int n) {
        if (n_alloc == n) return;
        n_alloc = n;
        calD1.resize(); calD2.resize();
        calD1_new.resize(); calD2_new.resize(); X_new.resize();
    }
};
static thread_local FwdEnvCEBufs s_fwd_ce_bufs;

static void forward_environment_ce(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env) {

    s_fwd_ce_bufs.ensure(g_n);
    auto& calD1 = s_fwd_ce_bufs.calD1;
    auto& calD2 = s_fwd_ce_bufs.calD2;
    auto& calD1_new = s_fwd_ce_bufs.calD1_new;
    auto& calD2_new = s_fwd_ce_bufs.calD2_new;
    auto& X_new = s_fwd_ce_bufs.X_new;

    const int TRI = g_n * (g_n + 1) / 2;
    for (int i = 0; i < TRI; ++i) {
        calD1.data[i] = Pi_1 * D1.data[i];
        calD2.data[i] = Pi_2 * D2.data[i];
    }

    env.X.resize();
    state_kernel_from_calD(calD1, calD2, env.X);

    for (int it = 0; it < inner_iters; ++it) {
        #pragma omp parallel sections
        {
            #pragma omp section
            compute_ce_filter_and_calD(env.X, D1, obs_idx_1, obs_gain1,
                                        env.Xtilde1, calD1_new);
            #pragma omp section
            compute_ce_filter_and_calD(env.X, D2, obs_idx_2, obs_gain2,
                                        env.Xtilde2, calD2_new);
        }

        state_kernel_from_calD(calD1_new, calD2_new, X_new);
        for (int i = 0; i < TRI; ++i)
            env.X.data[i] = 0.6 * X_new.data[i] + 0.4 * env.X.data[i];
    }

    env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
    env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
    // CE mode uses exact discrete projection; no Kalman-gain attenuation.
    env.scale1.fill(1.0); env.scale2.fill(1.0);
}

// --- forward_environment ---

// Thread-local reusable buffers for forward_environment to avoid
// repeated Kernel2D heap allocation across Picard iterations.
struct FwdEnvBufs {
    Kernel2D calD1, calD2, calD1_new, calD2_new, X_new;
    int n_alloc = 0;
    void ensure(int n) {
        if (n_alloc == n) return;
        n_alloc = n;
        calD1.resize(); calD2.resize();
        calD1_new.resize(); calD2_new.resize(); X_new.resize();
    }
};
static thread_local FwdEnvBufs s_fwd_bufs;

void forward_environment(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env) {

    s_fwd_bufs.ensure(g_n);
    auto& calD1 = s_fwd_bufs.calD1;
    auto& calD2 = s_fwd_bufs.calD2;
    auto& calD1_new = s_fwd_bufs.calD1_new;
    auto& calD2_new = s_fwd_bufs.calD2_new;
    auto& X_new = s_fwd_bufs.X_new;

    const int TRI = g_n * (g_n + 1) / 2;
    for (int i = 0; i < TRI; ++i) {
        calD1.data[i] = Pi_1 * D1.data[i];
        calD2.data[i] = Pi_2 * D2.data[i];
    }

    // Work directly in env.X to avoid final copy
    env.X.resize();
    state_kernel_from_calD(calD1, calD2, env.X);

    for (int it = 0; it < inner_iters; ++it) {
        #pragma omp parallel sections
        {
            #pragma omp section
            compute_filter_kernels(env.X, Pi_1, obs_idx_1, obs_gain1,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   env.Xtilde1, nullptr, &env.scale1);
            #pragma omp section
            compute_filter_kernels(env.X, Pi_2, obs_idx_2, obs_gain2,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   env.Xtilde2, nullptr, &env.scale2);
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            primitive_control_kernel(D1, env.Xtilde1,
                                     obs_gain1, obs_idx_1, Pi_1, calD1_new);
            #pragma omp section
            primitive_control_kernel(D2, env.Xtilde2,
                                     obs_gain2, obs_idx_2, Pi_2, calD2_new);
        }

        state_kernel_from_calD(calD1_new, calD2_new, X_new);
        for (int i = 0; i < TRI; ++i)
            env.X.data[i] = 0.6 * X_new.data[i] + 0.4 * env.X.data[i];
    }

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
                      Kernel2D& Hx,
                      const std::array<double, N_MAX>* scale_k) {
    Hx.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = -terminal_state_weight * X[g_n - 1][s];

    ensure_hk_buffers();
    for (auto& m : s_hk_buf0->data) m.setZero();
    HkSlice* cur = s_hk_buf0.get();
    HkSlice* nxt = s_hk_buf1.get();

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;  // t+1
        // Attenuate manipulation pricing by the discrete Kalman gain so it
        // matches the actual forward filter sensitivity (not the continuous
        // linearization). Defaults to 1.0 when scale_k is not provided.
        const double scale_tp = scale_k ? (*scale_k)[tp] : 1.0;
        const double dt_Pk = g_dt * prec_k[tp] * scale_tp;

        // Precompute Xtilde row pointer for tp
        const Vec3* Xt_tp = &Xtildek[tp][0];

        // W[r] = dt_Pk * sum_z Hk[tp][z][r]^T * Xtilde[tp][z]
        std::array<Vec3, N_MAX> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += (*cur)(z, r).transpose() * Xt_tp[z];
            W[r] = dt_Pk * acc;
        }

        // Precompute row pointers for tp
        const Vec3* Hx_tp = &Hx[tp][0];
        const Vec3* X_tp = &X[tp][0];
        const Vec3* Dk_tp = &Dk[tp][0];

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx_tp[r] + g_dt * (X_tp[r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                (*nxt)(z, r) = (*cur)(z, r)
                    + g_dt * (Dk_tp[r] * Hx_tp[z].transpose()
                          - X_tp[z] * W[r].transpose());

        std::swap(cur, nxt);
    }
}

// Version that also outputs the kernel information wedge V^i(t,r).
// V[t][r] = g_dt * Pk * sum_z Hk[t][z][r]^T * Xtilde[t][z]
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N_MAX>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel2D& Vkernel,
                      const std::array<double, N_MAX>* scale_k) {
    Hx.setZero();
    Vkernel.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = -terminal_state_weight * X[g_n - 1][s];

    ensure_hk_buffers();
    for (auto& m : s_hk_buf0->data) m.setZero();
    HkSlice* cur = s_hk_buf0.get();
    HkSlice* nxt = s_hk_buf1.get();

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        const double scale_tp = scale_k ? (*scale_k)[tp] : 1.0;
        const double dt_Pk = g_dt * prec_k[tp] * scale_tp;
        const Vec3* Xt_tp = &Xtildek[tp][0];

        std::array<Vec3, N_MAX> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += (*cur)(z, r).transpose() * Xt_tp[z];
            W[r] = dt_Pk * acc;
        }

        for (int r = 0; r <= j; ++r)
            Vkernel[tp][r] = W[r];

        const Vec3* Hx_tp = &Hx[tp][0];
        const Vec3* X_tp = &X[tp][0];
        const Vec3* Dk_tp = &Dk[tp][0];

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx_tp[r] + g_dt * (X_tp[r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                (*nxt)(z, r) = (*cur)(z, r)
                    + g_dt * (Dk_tp[r] * Hx_tp[z].transpose()
                          - X_tp[z] * W[r].transpose());

        std::swap(cur, nxt);
    }
}

// Legacy version that also fills Hk (for figure output)
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N_MAX>& prec_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel3D& Hk,
                      const std::array<double, N_MAX>* scale_k) {
    Hx.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = -terminal_state_weight * X[g_n - 1][s];

    for (int z = 0; z < g_n; ++z)
        for (int r = 0; r < g_n; ++r)
            Hk[g_n - 1][z][r].setZero();

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];
        const double scale_tp = scale_k ? (*scale_k)[tp] : 1.0;

        std::array<Vec3, N_MAX> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += Hk[tp][z][r].transpose() * Xtildek[tp][z];
            W[r] = g_dt * Pk * scale_tp * acc;
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
    const std::array<double, N_MAX>& prec_k, double terminal_weight,
    const std::array<double, N_MAX>* scale_k) {

    BackwardBarResult res;
    res.barHx.fill(0.0);
    res.barHk.setZero();
    res.barHx[g_n - 1] = terminal_weight * (barX[g_n - 1] - b);

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];
        const double scale_tp = scale_k ? (*scale_k)[tp] : 1.0;

        double I_val = 0.0;
        for (int z = 0; z <= tp; ++z)
            I_val += Xtildek[tp][z].dot(res.barHk[tp][z]);
        I_val *= g_dt * Pk * scale_tp;

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
        barX[0] = g_x0;
        for (int j = 0; j < g_n - 1; ++j)
            barX[j + 1] = barX[j] + g_dt * (barD1[j] + barD2[j]);

        BackwardBarResult bba1, bba2;
        #pragma omp parallel sections
        {
            #pragma omp section
            bba1 = backward_bar_adjoints(env.X, env.Xtilde2, D2,
                                          barX, g_b1, prec2_arr, 0.0, &env.scale2);
            #pragma omp section
            bba2 = backward_bar_adjoints(env.X, env.Xtilde1, D1,
                                          barX, g_b2, prec1_arr, 0.0, &env.scale1);
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
    sol.barX[0] = g_x0;
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
    const Mat3& Pi_2, int obs_idx_2,
    int fwd_inner_iters = FORWARD_INNER_ITERS,
    bool use_aa = true,
    int aa_depth = 5,
    int max_iters = MAX_PICARD_ITERS) {

    std::vector<double> residuals;
    auto prec1 = make_constant_prec(p1_val * p1_val);
    auto prec2 = make_constant_prec(p2_val * p2_val);
    EnvironmentResult env;
    const int TRI = g_n * (g_n + 1) / 2;
    const double neg_inv_r1 = -(1.0 / g_r1);
    const double neg_inv_r2 = -(1.0 / g_r2);

    // Pre-allocate loop temporaries once (avoid per-iteration heap alloc)
    Kernel2D Hx1, Hx2;

    // Adaptive relaxation: the Picard operator's spectral radius grows as
    // ~1/min(r1,r2), so the maximum stable relaxation shrinks for extreme
    // cost asymmetry. We start with a heuristic estimate and halve when
    // the residual grows for consecutive iterations.
    double min_r = std::min(g_r1, g_r2);
    double relax = std::min(PICARD_RELAX, 0.42 / (1.0 + 0.2 / min_r));
    double prev_err = 1e30;
    int grow_count = 0;           // consecutive iterations where residual grew
    int backoff_count = 0;        // total number of backoffs
    constexpr int GROW_LIMIT = 2; // halve relax after this many
    constexpr double RELAX_MIN = 0.005;

    // Anderson acceleration (AA) state — only allocated when needed.
    // AA extrapolation can destabilize for high spectral radius (extreme cost
    // asymmetry). When adaptive backoff triggers, we disable AA and fall back
    // to plain Picard which is always stable at sufficiently small alpha.
    constexpr double AA_REG = 1e-12;
    double aa_beta = std::min(0.6, 4.0 * relax); // AA mixing tracks relaxation
    bool aa_active = use_aa; // can be disabled dynamically by adaptive backoff
    const int aa_m = use_aa ? aa_depth : 0;
    const int aa_store = aa_m + 1;
    struct AAEntry { Kernel2D f1, f2, g1, g2; };
    std::vector<AAEntry> hist(use_aa ? aa_store : 0);
    // Picard path: single-entry residual buffer (no g1/g2 needed)
    Kernel2D picard_f1, picard_f2;
    int stored = 0;

    for (int it = 1; it <= max_iters; ++it) {
        forward_environment(D1, D2, p1_val, p2_val, fwd_inner_iters,
                            Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);

        #pragma omp parallel sections
        {
            #pragma omp section
            backward_kernels(env.X, env.Xtilde2, D2, prec2, 0.0, Hx1, &env.scale2);
            #pragma omp section
            backward_kernels(env.X, env.Xtilde1, D1, prec1, 0.0, Hx2, &env.scale1);
        }

        // Compute residual and update D
        double norm_f = 0.0, norm_g = 0.0;

        if (use_aa) {
            auto& cur = hist[stored % aa_store];
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
                std::cout << "  it=" << it << "  resid=" << err
                          << "  relax=" << relax << std::endl;
            if (err < PICARD_TOL) {
                if (verbose) std::cout << "  Converged at iteration " << it << std::endl;
                break;
            }

            // Adaptive relaxation: back off when residual grows
            if (err >= prev_err) {
                if (++grow_count >= GROW_LIMIT) {
                    ++backoff_count;
                    if (aa_active && backoff_count >= 2) {
                        // AA extrapolation is destabilizing — switch to plain
                        // Picard which is always stable at small enough alpha.
                        // Reset D to zero so we don't carry over chaotic AA state.
                        aa_active = false;
                        relax = std::min(PICARD_RELAX, 0.42 / (1.0 + 0.2 / min_r));
                        D1.setZero();
                        D2.setZero();
                        prev_err = 1e30;
                        if (verbose)
                            std::cout << "  [adaptive] AA disabled, cold restart, relax=" << relax << std::endl;
                    } else if (relax > RELAX_MIN) {
                        relax = std::max(RELAX_MIN, relax * 0.5);
                        aa_beta = std::min(aa_beta, 2.0 * relax);
                        stored = 0;  // reset AA history after relaxation change
                        if (verbose)
                            std::cout << "  [adaptive] relax -> " << relax << std::endl;
                    }
                    grow_count = 0;
                }
            } else {
                grow_count = 0;
            }
            prev_err = err;

            int m = aa_active ? std::min(stored, aa_m) : 0;
            if (m >= 1) {
                Eigen::VectorXd Ffi(m);
                for (int i = 0; i < m; ++i) {
                    int idx = (stored - 1 - i + aa_store) % aa_store;
                    Ffi(i) = kernel_dot(cur.f1, cur.f2, hist[idx].f1, hist[idx].f2);
                }
                Eigen::MatrixXd H(m, m);
                Eigen::VectorXd rhs(m);
                for (int i = 0; i < m; ++i) {
                    rhs(i) = norm_f - Ffi(i);
                    for (int j = 0; j <= i; ++j) {
                        int ii = (stored - 1 - i + aa_store) % aa_store;
                        int jj = (stored - 1 - j + aa_store) % aa_store;
                        double fifj = kernel_dot(hist[ii].f1, hist[ii].f2, hist[jj].f1, hist[jj].f2);
                        H(i, j) = H(j, i) = norm_f - Ffi(i) - Ffi(j) + fifj;
                    }
                }
                for (int i = 0; i < m; ++i)
                    H(i, i) += AA_REG * (1.0 + H(i, i));
                Eigen::VectorXd gamma = H.ldlt().solve(rhs);

                std::vector<const Vec3*> g1p(m), g2p(m);
                std::vector<double> gam(m);
                for (int i = 0; i < m; ++i) {
                    int idx = (stored - 1 - i + aa_store) % aa_store;
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
                    D1.data[i] = (1.0 - aa_beta) * D1.data[i] + aa_beta * d1_aa;
                    D2.data[i] = (1.0 - aa_beta) * D2.data[i] + aa_beta * d2_aa;
                }
            } else {
                for (int i = 0; i < TRI; ++i) {
                    D1.data[i] += relax * cur.f1.data[i];
                    D2.data[i] += relax * cur.f2.data[i];
                }
            }
            if (aa_active) stored++;
        } else {
            // Picard: compute residual directly, no g1/g2 storage needed
            for (int i = 0; i < TRI; ++i) {
                Vec3 g1i = neg_inv_r1 * Hx1.data[i];
                Vec3 g2i = neg_inv_r2 * Hx2.data[i];
                picard_f1.data[i] = g1i - D1.data[i];
                picard_f2.data[i] = g2i - D2.data[i];
                norm_f += picard_f1.data[i].squaredNorm() + picard_f2.data[i].squaredNorm();
                norm_g += g1i.squaredNorm() + g2i.squaredNorm();
            }

            double err = std::sqrt(norm_f) / std::max(1.0, std::sqrt(norm_g));
            residuals.push_back(err);
            if (!std::isfinite(err)) break;
            if (verbose && (it <= 5 || it % 50 == 0))
                std::cout << "  it=" << it << "  resid=" << err
                          << "  relax=" << relax << std::endl;
            if (err < PICARD_TOL) {
                if (verbose) std::cout << "  Converged at iteration " << it << std::endl;
                break;
            }

            // Adaptive relaxation: back off when residual grows
            if (err >= prev_err) {
                if (++grow_count >= GROW_LIMIT && relax > RELAX_MIN) {
                    relax = std::max(RELAX_MIN, relax * 0.5);
                    grow_count = 0;
                    if (verbose)
                        std::cout << "  [adaptive] relax -> " << relax << std::endl;
                }
            } else {
                grow_count = 0;
            }
            prev_err = err;

            for (int i = 0; i < TRI; ++i) {
                D1.data[i] += relax * picard_f1.data[i];
                D2.data[i] += relax * picard_f2.data[i];
            }
        }
    }

    // Convergence check breaks before Anderson update, so env matches D1/D2
    // Derive A_store from existing Xtilde for external use (materialize_F, etc.)
    {
        double dt_prec1 = g_dt * p1_val * p1_val;
        double dt_prec2 = g_dt * p2_val * p2_val;
        env.A_store1[0][0].setZero();
        env.A_store2[0][0].setZero();
        for (int j = 1; j < g_n; ++j)
            for (int s = 0; s < j; ++s) {
                env.A_store1[j][s] = dt_prec1 * env.Xtilde1[j][s];
                env.A_store2[j][s] = dt_prec2 * env.Xtilde2[j][s];
            }
    }

    Kernel2D calD1, calD2;
    #pragma omp parallel sections
    {
        #pragma omp section
        primitive_control_kernel(D1, env.Xtilde1,
                                 p1_val, obs_idx_1, Pi_1, calD1);
        #pragma omp section
        primitive_control_kernel(D2, env.Xtilde2,
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
    const Mat3& Pi_2, int obs_idx_2,
    int max_iters) {
    Kernel2D D1(D1_init), D2(D2_init);
    return solve_equilibrium_core(p1_val, p2_val, std::move(D1), std::move(D2),
                                  verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2,
                                  FORWARD_INNER_ITERS, true, 5, max_iters);
}

// --- solve_equilibrium_fast: memory-efficient Picard solver ---
//
// Uses S_pi warm start from the costate decomposition Hx = S·X + R:
//   - Pure Picard iteration (no Anderson acceleration)
//   - No AA history buffer needed (saves 24 Kernel2D = ~7.2 MB at N=160)
//   - Same fixed point as standard solver (D_gap = 0)
//   - Falls back to standard AA solver if Picard doesn't converge

EquilibriumResult solve_equilibrium_fast(
    double p1_val, double p2_val, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {

    // S_pi warm start
    std::array<double, N_MAX> S_pi;
    S_pi.fill(0.0);
    S_pi[g_n - 1] = TERMINAL_STATE_WEIGHT;
    for (int j = g_n - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1]
            + g_dt * (1.0 - (1.0 / g_r1 + 1.0 / g_r2) * S_pi[j + 1] * S_pi[j + 1]);

    double r_ratio = std::min(g_r1, g_r2) / std::max(g_r1, g_r2);
    double damp = std::sqrt(r_ratio);

    Kernel2D D1, D2;
    Vec3 sigE0 = g_sigma * E0();
    for (int j = 0; j < g_n; ++j)
        for (int s = 0; s <= j; ++s) {
            D1[j][s] = -damp * (S_pi[j] / g_r1) * sigE0;
            D2[j][s] = -damp * (S_pi[j] / g_r2) * sigE0;
        }

    // Pure Picard iteration (no Anderson acceleration).
    // S_pi warm start places us close enough that plain relaxation converges,
    // and we avoid all AA history memory (saves 24 Kernel2D = ~460 KB at N=40).
    auto result = solve_equilibrium_core(p1_val, p2_val, std::move(D1), std::move(D2),
                                          verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2,
                                          FORWARD_INNER_ITERS, /*use_aa=*/false);

    // Fallback to standard solver if warm start didn't converge or diverged.
    // Note: NaN >= x is false in IEEE 754, so check !isfinite explicitly.
    if (!result.residuals.empty() &&
        (!std::isfinite(result.residuals.back()) ||
         result.residuals.back() >= PICARD_TOL)) {
        if (verbose)
            std::cout << "  [fast] Did not converge; falling back to standard\n";
        return solve_equilibrium(p1_val, p2_val, verbose,
                                 Pi_1, obs_idx_1, Pi_2, obs_idx_2);
    }

    return result;
}

// --- solve_equilibrium_sr: Riccati warm-started Picard iteration ---
//
// Uses the perfect-info scalar Riccati S_pi to initialize D, giving
// Anderson acceleration a much better starting point than D=0.
// Falls back to standard cold start if the warm start doesn't converge.

EquilibriumResult solve_equilibrium_sr(
    double p1_val, double p2_val, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {

    // Perfect-info scalar Riccati
    std::array<double, N_MAX> S_pi;
    S_pi.fill(0.0);
    S_pi[g_n - 1] = TERMINAL_STATE_WEIGHT;
    for (int j = g_n - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1]
            + g_dt * (1.0 - (1.0 / g_r1 + 1.0 / g_r2) * S_pi[j + 1] * S_pi[j + 1]);

    // Initialize D from S_pi applied to the uncontrolled state kernel (sigma·E0).
    // Damped by cost asymmetry: when r1 ≈ r2, the joint Riccati S_pi is a good
    // approximation to both players' equilibrium S, so we use the full warm start.
    // When costs are very asymmetric, S_pi doesn't approximate either player's S
    // well, so we damp toward D=0 to avoid misleading Anderson acceleration.
    double r_ratio = std::min(g_r1, g_r2) / std::max(g_r1, g_r2);  // in (0,1]
    double damp = std::sqrt(r_ratio);  // 1.0 for symmetric, ~0.5 for 4:1 ratio

    Kernel2D D1, D2;
    Vec3 sigE0 = g_sigma * E0();
    for (int j = 0; j < g_n; ++j)
        for (int s = 0; s <= j; ++s) {
            D1[j][s] = -damp * (S_pi[j] / g_r1) * sigE0;
            D2[j][s] = -damp * (S_pi[j] / g_r2) * sigE0;
        }

    auto result = solve_equilibrium_core(p1_val, p2_val, std::move(D1), std::move(D2),
                                          verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2);

    // Fallback: if warm start didn't converge or diverged, retry from cold start
    if (!result.residuals.empty() &&
        (!std::isfinite(result.residuals.back()) ||
         result.residuals.back() >= PICARD_TOL)) {
        if (verbose)
            std::cout << "  [SR] Warm start did not converge; retrying cold start\n";
        Kernel2D D1z, D2z;
        D1z.setZero();
        D2z.setZero();
        result = solve_equilibrium_core(p1_val, p2_val, std::move(D1z), std::move(D2z),
                                         verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2);
    }

    return result;
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

    // Hx buffers hoisted out of loop (avoids 309KB alloc per iteration at N=160)
    Kernel2D Hx1, Hx2;

    for (int it = 1; it <= MAX_PICARD_ITERS; ++it) {
        // Forward pass using discrete CE
        forward_environment_ce(D1, D2, p1_val, p2_val, FORWARD_INNER_ITERS,
                               Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);

        // Backward pass (uses Xtilde, same as standard solver)
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

    // After convergence: populate A_store (for F materialization / API compat)
    // and compute final calD via CE projection, all in parallel.
    // Reuse Hx1/Hx2 as scratch for the throwaway Xtilde from compute_filter_kernels.
    Kernel2D calD1, calD2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            compute_filter_kernels(env.X, Pi_1, obs_idx_1, p1_val,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   Hx1, &env.A_store1);
            compute_ce_filter_and_calD(env.X, D1, obs_idx_1, p1_val,
                                        env.Xtilde1, calD1);
        }
        #pragma omp section
        {
            compute_filter_kernels(env.X, Pi_2, obs_idx_2, p2_val,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   Hx2, &env.A_store2);
            compute_ce_filter_and_calD(env.X, D2, obs_idx_2, p2_val,
                                        env.Xtilde2, calD2);
        }
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

std::unique_ptr<FSlice> compute_ce_F_slice_at_T(
    const Kernel2D& X, int obs_idx, double obs_gain, const Mat3& Pi) {

    int n = g_n;
    int dim = 3 * n;
    int n_obs = n - 1;
    double g = obs_gain;

    // Build thin V basis incrementally
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

    // Extract F: F(u,s) = [VV^T(3u:3u+3, 3s:3s+3) - Pi*delta(u,s)] / dt
    auto F = std::make_unique<FSlice>();
    double inv_dt = 1.0 / g_dt;
    auto Vr = V.leftCols(rank);
    for (int u = 0; u < n; ++u) {
        auto Vu = Vr.middleRows(3*u, 3);
        for (int s = 0; s < n; ++s) {
            auto Vs = Vr.middleRows(3*s, 3);
            Mat3 Mus = Vu * Vs.transpose();
            if (u == s) Mus -= Pi;
            (*F)(u, s) = Mus * inv_dt;
        }
    }

    return F;
}

ProjectionPair exact_discrete_CE(const EquilibriumResult& eq) {
    int t_idx = g_n - 1;
    DiscreteProjection p1, p2;
    #pragma omp parallel sections
    {
        #pragma omp section
        p1 = discrete_conditional_expectation(eq.env.X, eq.env.obs_idx1, eq.env.obs_gain1, t_idx);
        #pragma omp section
        p2 = discrete_conditional_expectation(eq.env.X, eq.env.obs_idx2, eq.env.obs_gain2, t_idx);
    }
    return {std::move(p1), std::move(p2)};
}

// --- S/R costate decomposition ---
//
// At each time j, project Hx[j][·] onto X[j][·] to extract the scalar
// Riccati coefficient S[j].  The residual R = Hx - S·X is orthogonal to X
// (in the ℓ² inner product over the source index r).

CostateDecomposition decompose_costate(
    const Kernel2D& Hx, const Kernel2D& X, const Kernel2D& Vkernel) {

    CostateDecomposition d;
    d.R.setZero();
    d.V_R.setZero();
    d.S.fill(0.0);
    d.V_S.fill(0.0);

    double Hx_sq = 0.0, R_sq = 0.0;
    double V_sq = 0.0, VR_sq = 0.0;

    for (int j = 0; j < g_n; ++j) {
        // S[j] = <Hx[j], X[j]> / <X[j], X[j]>
        double hx_x = 0.0, x_x = 0.0;
        for (int r = 0; r <= j; ++r) {
            hx_x += Hx[j][r].dot(X[j][r]);
            x_x  += X[j][r].squaredNorm();
        }
        d.S[j] = (x_x > 1e-30) ? hx_x / x_x : 0.0;

        // R[j][r] = Hx[j][r] - S[j]·X[j][r]
        for (int r = 0; r <= j; ++r) {
            d.R[j][r] = Hx[j][r] - d.S[j] * X[j][r];
            Hx_sq += Hx[j][r].squaredNorm();
            R_sq  += d.R[j][r].squaredNorm();
        }

        // V_S[j] = <V[j], X[j]> / <X[j], X[j]>
        double v_x = 0.0;
        for (int r = 0; r <= j; ++r)
            v_x += Vkernel[j][r].dot(X[j][r]);
        d.V_S[j] = (x_x > 1e-30) ? v_x / x_x : 0.0;

        for (int r = 0; r <= j; ++r) {
            d.V_R[j][r] = Vkernel[j][r] - d.V_S[j] * X[j][r];
            V_sq  += Vkernel[j][r].squaredNorm();
            VR_sq += d.V_R[j][r].squaredNorm();
        }
    }

    d.R_frac   = (Hx_sq > 1e-30) ? std::sqrt(R_sq / Hx_sq) : 0.0;
    d.V_R_frac = (V_sq  > 1e-30) ? std::sqrt(VR_sq / V_sq)  : 0.0;

    return d;
}

void backward_kernels_sr(
    const Kernel2D& X, const Kernel2D& Xtildek,
    const Kernel2D& Dk, const std::array<double, N_MAX>& prec_k,
    double terminal_state_weight,
    Kernel2D& Hx, Kernel2D& Vkernel,
    CostateDecomposition& decomp) {

    // Standard backward pass (unchanged)
    backward_kernels(X, Xtildek, Dk, prec_k, terminal_state_weight, Hx, Vkernel);

    // Post-process: extract S/R decomposition
    decomp = decompose_costate(Hx, X, Vkernel);
}
