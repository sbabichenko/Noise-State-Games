// WASM entry point for the interactive LQG solver.
// Compiled with Emscripten, called from JavaScript via ccall/cwrap.

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

static void out_kernel2d(const char* name, const Kernel2D& K) {
    char buf[32];
    g_output += '"'; g_output += name; g_output += "\":{";
    for (int ch = 0; ch < 3; ++ch) {
        if (ch) g_output += ',';
        g_output += "\"ch"; g_output += ('0' + ch); g_output += "\":[";
        for (int ti = 0; ti < g_n; ++ti) {
            if (ti) g_output += ',';
            g_output += '[';
            for (int si = 0; si <= ti; ++si) {
                if (si) g_output += ',';
                int n = snprintf(buf, sizeof(buf), "%.6g", K[ti][si](ch));
                g_output.append(buf, n);
            }
            for (int si = ti + 1; si < g_n; ++si)
                g_output += ",0";
            g_output += ']';
        }
        g_output += ']';
    }
    g_output += '}';
}

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

// Serialize an FSlice as a 3x3 grid of 2D arrays: F_row_col[u][s]
static void out_fslice(const char* name, const FSlice& F) {
    char buf[32];
    g_output += '"'; g_output += name; g_output += "\":{";
    bool first = true;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            if (!first) g_output += ',';
            first = false;
            // Key: "r0c0", "r0c1", etc.
            g_output += "\"r"; g_output += ('0' + row);
            g_output += "c"; g_output += ('0' + col); g_output += "\":[";
            for (int u = 0; u < g_n; ++u) {
                if (u) g_output += ',';
                g_output += '[';
                for (int s = 0; s < g_n; ++s) {
                    if (s) g_output += ',';
                    int n = snprintf(buf, sizeof(buf), "%.6g", F(u, s)(row, col));
                    g_output.append(buf, n);
                }
                g_output += ']';
            }
            g_output += ']';
        }
    }
    g_output += '}';
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void wasm_set_grid(int n, double T) {
    set_grid(n, T);
}

// Returns pointer to JSON string (owned by g_output, valid until next call)
EMSCRIPTEN_KEEPALIVE
const char* solve_single(double p1, double p2, double b1, double b2, double r1, double r2) {
    g_output.clear();
    g_output.reserve(64 * 1024);
    g_b1 = b1; g_b2 = b2;
    g_r1 = r1; g_r2 = r2;

    auto eq = solve_equilibrium(p1, p2, false);
    auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                      p1*p1, p2*p2, 2000, 0.08, 1e-10);
    auto costs = compute_costs_general(eq.env, eq.calD1, eq.calD2, bar, r1, r2, b1, b2);

    auto prec1_arr = make_constant_prec(p1 * p1);
    auto prec2_arr = make_constant_prec(p2 * p2);
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

    std::array<double, N_MAX> barD1_pi, barX_pi;
    compute_perfect_info(b1, b2, barD1_pi, barX_pi);

    const auto& tg = t_grid();
    out("{");
    out_array("t", tg.data(), g_n);
    out(",\"residuals\":[");
    for (size_t i = 0; i < eq.residuals.size(); ++i)
        out("%s%.6g", i ? "," : "", eq.residuals[i]);
    out("],\"n_iters\":%zu", eq.residuals.size());

    out(","); out_kernel2d("X", eq.env.X);
    out(","); out_kernel2d("D1", eq.D1);
    out(","); out_kernel2d("D2", eq.D2);
    out(","); out_kernel2d("calD1", eq.calD1);

    out(","); out_array("barD1", bar.barD1.data(), g_n);
    out(","); out_array("barD2", bar.barD2.data(), g_n);
    out(","); out_array("barX", bar.barX.data(), g_n);
    out(",\"bar_residual\":%.6g", bar.bar_residual);
    out(","); out_array("V1", V1_arr.data(), g_n);
    out(","); out_array("V2", V2_arr.data(), g_n);
    out(","); out_array("barD1_pi", barD1_pi.data(), g_n);
    out(","); out_array("barX_pi", barX_pi.data(), g_n);
    out(",\"J1\":%.6g,\"J2\":%.6g", costs.J1, costs.J2);
    out(",\"p1\":%.6g,\"p2\":%.6g,\"b1\":%.6g,\"b2\":%.6g", p1, p2, b1, b2);
    out("}");

    return g_output.c_str();
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

// Compute F kernel slices at t=T for both players. Returns JSON with F1 and F2.
EMSCRIPTEN_KEEPALIVE
const char* solve_f_kernel(double p1, double p2, double b1, double b2, double r1, double r2) {
    g_output.clear();
    g_output.reserve(128 * 1024);
    g_b1 = b1; g_b2 = b2;
    g_r1 = r1; g_r2 = r2;

    auto eq = solve_equilibrium(p1, p2, false);

    FSlice F1 = compute_F_slice_at_T(eq.env.Xtilde1, eq.env.A_store1,
                                      eq.env.obs_gain1, eq.env.obs_idx1);
    FSlice F2 = compute_F_slice_at_T(eq.env.Xtilde2, eq.env.A_store2,
                                      eq.env.obs_gain2, eq.env.obs_idx2);

    const auto& tg = t_grid();
    out("{");
    out_array("t", tg.data(), g_n);
    out(","); out_fslice("F1", F1);
    out(","); out_fslice("F2", F2);
    out("}");

    return g_output.c_str();
}

} // extern "C"
