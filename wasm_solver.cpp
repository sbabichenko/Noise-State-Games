// WASM entry point for the interactive LQG solver.
// Compiled with Emscripten, called from JavaScript via ccall/cwrap.
//
// Architecture: a C++-side cache holds the last EquilibriumResult, BarSolution,
// and derived quantities. Multiple WASM endpoints (solve_light, solve_full,
// solve_f_kernel, solve_sweep) all use ensure_solve() to avoid redundant
// recomputation when the same parameters are queried from different tabs.

#include "lqg_solver.h"
#include <emscripten/emscripten.h>
#include <cstdio>
#include <cstring>
#include <string>

// Shared output buffer (grows as needed)
static std::string g_output;

static void out(const char* fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    g_output += buf;
}

static void out_array(const char* name, const double* arr, int n) {
    out("\"%s\":[", name);
    for (int i = 0; i < n; ++i)
        out("%s%.6g", i ? "," : "", arr[i]);
    out("]");
}

// (out_kernel2d removed — kernel2d data now uses binary transfer via solve_full_bin)

static void compute_perfect_info(double b1, double b2,
                                  std::array<double, N_MAX>& barD1_pi,
                                  std::array<double, N_MAX>& barX_pi) {
    std::array<double, N_MAX> S_pi;
    S_pi.fill(0.0);
    S_pi[g_n - 1] = TERMINAL_STATE_WEIGHT;
    for (int j = g_n - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1] + g_dt * (1.0 - (1.0 / g_r1 + 1.0 / g_r2) * S_pi[j + 1] * S_pi[j + 1]);
    barX_pi[0] = X0;
    for (int j = 0; j < g_n; ++j) {
        barD1_pi[j] = -(1.0 / g_r1) * S_pi[j] * (barX_pi[j] - b1);
        double barD2_pi_j = -(1.0 / g_r2) * S_pi[j] * (barX_pi[j] - b2);
        if (j < g_n - 1)
            barX_pi[j + 1] = barX_pi[j] + g_dt * (barD1_pi[j] + barD2_pi_j);
    }
}

// (out_fslice removed — F kernel now uses binary transfer via solve_f_kernel_bin)

// ============================================================
// Solver result cache
// ============================================================
// Avoids redundant solve_equilibrium calls when switching between tabs
// (e.g. Equilibrium → Kernels → F Kernel all need the same solve).

static struct SolveCache {
    double p1, p2, b1, b2, r1, r2;
    int n; double T;
    bool valid;

    EquilibriumResult eq;
    BarSolution bar;
    CostPair costs;
    std::array<double, N_MAX> V1, V2;
    std::array<double, N_MAX> barD1_pi, barX_pi;

    // F kernel cache (computed lazily on first F kernel tab visit)
    std::unique_ptr<FSlice> F1, F2;
    bool f_valid;
} g_cache = {};

static bool cache_matches(double p1, double p2, double b1, double b2, double r1, double r2) {
    return g_cache.valid &&
           g_cache.p1 == p1 && g_cache.p2 == p2 &&
           g_cache.b1 == b1 && g_cache.b2 == b2 &&
           g_cache.r1 == r1 && g_cache.r2 == r2 &&
           g_cache.n == g_n && g_cache.T == g_T;
}

// Run the full solve pipeline if not cached. After this, g_cache holds all results.
static void ensure_solve(double p1, double p2, double b1, double b2, double r1, double r2) {
    if (cache_matches(p1, p2, b1, b2, r1, r2)) return;

    g_b1 = b1; g_b2 = b2;
    g_r1 = r1; g_r2 = r2;

    g_cache.eq = solve_equilibrium(p1, p2, false);
    g_cache.bar = solve_bar_equilibrium(g_cache.eq.env, g_cache.eq.D1, g_cache.eq.D2,
                                         p1*p1, p2*p2, 2000, 0.08, 1e-10);
    g_cache.costs = compute_costs_general(g_cache.eq.env, g_cache.eq.calD1, g_cache.eq.calD2,
                                           g_cache.bar, r1, r2, b1, b2);

    auto prec1_arr = make_constant_prec(p1 * p1);
    auto prec2_arr = make_constant_prec(p2 * p2);
    auto bba1 = backward_bar_adjoints(g_cache.eq.env.X, g_cache.eq.env.Xtilde2, g_cache.eq.D2,
                                       g_cache.bar.barX, b1, prec2_arr, 0.0);
    auto bba2 = backward_bar_adjoints(g_cache.eq.env.X, g_cache.eq.env.Xtilde1, g_cache.eq.D1,
                                       g_cache.bar.barX, b2, prec1_arr, 0.0);
    for (int j = 0; j < g_n; ++j) {
        double V1 = 0.0, V2 = 0.0;
        for (int z = 0; z <= j; ++z) {
            V1 += g_cache.eq.env.Xtilde2[j][z].dot(bba1.barHk[j][z]);
            V2 += g_cache.eq.env.Xtilde1[j][z].dot(bba2.barHk[j][z]);
        }
        g_cache.V1[j] = V1 * p2 * p2 * g_dt;
        g_cache.V2[j] = V2 * p1 * p1 * g_dt;
    }

    compute_perfect_info(b1, b2, g_cache.barD1_pi, g_cache.barX_pi);

    g_cache.p1 = p1; g_cache.p2 = p2;
    g_cache.b1 = b1; g_cache.b2 = b2;
    g_cache.r1 = r1; g_cache.r2 = r2;
    g_cache.n = g_n; g_cache.T = g_T;
    g_cache.valid = true;
    g_cache.f_valid = false;  // F kernel depends on equilibrium; recompute lazily
}

// Compute F kernel slices if not cached. Requires ensure_solve() first.
static void ensure_f_kernel(double p1, double p2, double b1, double b2, double r1, double r2) {
    ensure_solve(p1, p2, b1, b2, r1, r2);
    if (g_cache.f_valid) return;

    const auto& env = g_cache.eq.env;
    g_cache.F1 = compute_F_slice_at_T(env.Xtilde1, env.A_store1,
                                       env.obs_gain1, env.obs_idx1);
    g_cache.F2 = compute_F_slice_at_T(env.Xtilde2, env.A_store2,
                                       env.obs_gain2, env.obs_idx2);
    g_cache.f_valid = true;
}

// Serialize only the "light" data (bar solution, costs, wedges, residuals).
// ~5KB JSON. Used by Equilibrium and Diagnostics tabs.
static void serialize_light() {
    const auto& tg = t_grid();
    const auto& eq = g_cache.eq;
    const auto& bar = g_cache.bar;

    out("{");
    out_array("t", tg.data(), g_n);
    out(",\"residuals\":[");
    for (size_t i = 0; i < eq.residuals.size(); ++i)
        out("%s%.6g", i ? "," : "", eq.residuals[i]);
    out("],\"n_iters\":%zu", eq.residuals.size());

    out(","); out_array("barD1", bar.barD1.data(), g_n);
    out(","); out_array("barD2", bar.barD2.data(), g_n);
    out(","); out_array("barX", bar.barX.data(), g_n);
    out(",\"bar_residual\":%.6g", bar.bar_residual);
    out(","); out_array("V1", g_cache.V1.data(), g_n);
    out(","); out_array("V2", g_cache.V2.data(), g_n);
    out(","); out_array("barD1_pi", g_cache.barD1_pi.data(), g_n);
    out(","); out_array("barX_pi", g_cache.barX_pi.data(), g_n);
    out(",\"J1\":%.6g,\"J2\":%.6g", g_cache.costs.J1, g_cache.costs.J2);
    out(",\"p1\":%.6g,\"p2\":%.6g,\"b1\":%.6g,\"b2\":%.6g",
        g_cache.p1, g_cache.p2, g_cache.b1, g_cache.b2);
    out("}");
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void wasm_set_grid(int n, double T) {
    set_grid(n, T);
    g_cache.valid = false;  // grid change invalidates cache
    g_cache.f_valid = false;
}

// Light solve: bar solution + costs + wedges + residuals. ~5KB JSON.
// Used by Equilibrium tab and Diagnostics tab.
EMSCRIPTEN_KEEPALIVE
const char* solve_light(double p1, double p2, double b1, double b2, double r1, double r2) {
    g_output.clear();
    g_output.reserve(8 * 1024);
    ensure_solve(p1, p2, b1, b2, r1, r2);
    serialize_light();
    return g_output.c_str();
}

// Binary full solve: writes 4 kernels × 3 channels × N×N doubles to buffer.
// Layout: [X_ch0[N×N], X_ch1[N×N], X_ch2[N×N], D1_ch0, ..., calD1_ch2]
// Each channel is row-major lower-triangular (upper entries are 0).
// JS reads directly from HEAPF64 — no JSON serialize/parse.
static std::vector<double> g_full_buf;

EMSCRIPTEN_KEEPALIVE
double* solve_full_bin(double p1, double p2, double b1, double b2, double r1, double r2) {
    ensure_solve(p1, p2, b1, b2, r1, r2);

    const int n = g_n;
    const int block = n * n;
    // 4 kernels × 3 channels × N×N
    g_full_buf.resize(4 * 3 * block);

    const Kernel2D* kernels[4] = {
        &g_cache.eq.env.X, &g_cache.eq.D1, &g_cache.eq.D2, &g_cache.eq.calD1
    };

    for (int k = 0; k < 4; ++k) {
        const Kernel2D& K = *kernels[k];
        for (int ch = 0; ch < 3; ++ch) {
            double* dst = g_full_buf.data() + (k * 3 + ch) * block;
            for (int ti = 0; ti < n; ++ti) {
                for (int si = 0; si <= ti; ++si)
                    dst[ti * n + si] = K[ti][si](ch);
                for (int si = ti + 1; si < n; ++si)
                    dst[ti * n + si] = 0.0;
            }
        }
    }

    return g_full_buf.data();
}

EMSCRIPTEN_KEEPALIVE
const char* solve_sweep(double p1, double b1, double b2, double r1, double r2,
                         const double* p2_vals, int n_p2) {
    g_output.clear();
    g_output.reserve(32 * 1024);
    g_b1 = b1; g_b2 = b2;
    g_r1 = r1; g_r2 = r2;

    std::array<double, N_MAX> barD1_pi, barX_pi;
    compute_perfect_info(b1, b2, barD1_pi, barX_pi);

    const auto& tg = t_grid();
    out("{");
    out_array("t", tg.data(), g_n);
    out(","); out_array("barD1_pi", barD1_pi.data(), g_n);
    out(",\"p1\":%.6g,\"b1\":%.6g,\"b2\":%.6g", p1, b1, b2);

    out(",\"sweeps\":[");
    for (int i = 0; i < n_p2; ++i) {
        double p2 = p2_vals[i];
        if (i > 0) out(",");
        out("{\"p2\":%.6g", p2);

        auto eq = solve_equilibrium(p1, p2, false);
        auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                          p1*p1, p2*p2, 2000, 0.08, 1e-10);
        auto costs_priv = compute_costs_general(eq.env, eq.calD1, eq.calD2,
                                                 bar, r1, r2, b1, b2);
        out(","); out_array("barD1", bar.barD1.data(), g_n);
        out(",\"J1_priv\":%.6g,\"J2_priv\":%.6g", costs_priv.J1, costs_priv.J2);

        double p_common = std::sqrt(p1*p1 + p2*p2);
        auto eq_pool = solve_equilibrium(p_common, p_common, false,
                                          Pi1(), 1, Pi1(), 1);
        auto bar_pool = solve_bar_equilibrium(eq_pool.env, eq_pool.D1, eq_pool.D2,
                                               p_common*p_common, p_common*p_common,
                                               2000, 0.08, 1e-10);
        auto costs_pool = compute_costs_general(eq_pool.env, eq_pool.calD1, eq_pool.calD2,
                                                 bar_pool, r1, r2, b1, b2);
        out(",\"J1_pool\":%.6g,\"J2_pool\":%.6g", costs_pool.J1, costs_pool.J2);

        auto prec2_arr = make_constant_prec(p2 * p2);
        auto prec1_arr = make_constant_prec(p1 * p1);
        auto bba1 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde2, eq.D2,
                                           bar.barX, b1, prec2_arr, 0.0);
        auto bba2 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde1, eq.D1,
                                           bar.barX, b2, prec1_arr, 0.0);
        std::array<double, N_MAX> V1_arr, V2_arr;
        for (int j = 0; j < g_n; ++j) {
            double V1 = 0.0, V2 = 0.0;
            for (int z = 0; z <= j; ++z) {
                V1 += eq.env.Xtilde2[j][z].dot(bba1.barHk[j][z]);
                V2 += eq.env.Xtilde1[j][z].dot(bba2.barHk[j][z]);
            }
            V1_arr[j] = V1 * p2 * p2 * g_dt;
            V2_arr[j] = V2 * p1 * p1 * g_dt;
        }
        out(","); out_array("V1", V1_arr.data(), g_n);
        out(","); out_array("V2", V2_arr.data(), g_n);
        out("}");
    }
    out("]}");

    return g_output.c_str();
}

// Binary F kernel: writes 2 * 9 * g_n * g_n doubles to a static buffer.
// Layout: [F1_r0c0[0..n*n], F1_r0c1[..], ..., F1_r2c2[..], F2_r0c0[..], ..., F2_r2c2[..]]
// Each block is row-major: F[u*n + s] for u=0..n-1, s=0..n-1.
// JS reads directly from HEAPF64 — no JSON serialization or parsing.
static std::vector<double> g_fkernel_buf;

EMSCRIPTEN_KEEPALIVE
double* solve_f_kernel_bin(double p1, double p2, double b1, double b2, double r1, double r2) {
    ensure_f_kernel(p1, p2, b1, b2, r1, r2);  // cached — no re-solve if params match

    const int n = g_n;
    const int block = n * n;
    g_fkernel_buf.resize(2 * 9 * block);

    // Write F1 then F2
    const FSlice* slices[2] = { g_cache.F1.get(), g_cache.F2.get() };
    for (int p = 0; p < 2; ++p) {
        const FSlice& F = *slices[p];
        double* base = g_fkernel_buf.data() + p * 9 * block;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                double* dst = base + (r * 3 + c) * block;
                for (int u = 0; u < n; ++u)
                    for (int s = 0; s < n; ++s)
                        dst[u * n + s] = F.data[u][s](r, c);
            }
        }
    }

    return g_fkernel_buf.data();
}

} // extern "C"
