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
        for (int ti = 0; ti < N; ++ti) {
            if (ti) g_output += ',';
            g_output += '[';
            for (int si = 0; si <= ti; ++si) {
                if (si) g_output += ',';
                int n = snprintf(buf, sizeof(buf), "%.6g", K[ti][si](ch));
                g_output.append(buf, n);
            }
            for (int si = ti + 1; si < N; ++si)
                g_output += ",0";
            g_output += ']';
        }
        g_output += ']';
    }
    g_output += '}';
}

static void compute_perfect_info(double b1, double b2,
                                  std::array<double, N>& barD1_pi,
                                  std::array<double, N>& barX_pi) {
    std::array<double, N> S_pi;
    S_pi.fill(0.0);
    S_pi[N - 1] = TERMINAL_STATE_WEIGHT;
    for (int j = N - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1] + DT * (1.0 - (1.0 / g_r1 + 1.0 / g_r2) * S_pi[j + 1] * S_pi[j + 1]);
    barX_pi[0] = X0;
    for (int j = 0; j < N; ++j) {
        barD1_pi[j] = -(1.0 / g_r1) * S_pi[j] * (barX_pi[j] - b1);
        double barD2_pi_j = -(1.0 / g_r2) * S_pi[j] * (barX_pi[j] - b2);
        if (j < N - 1)
            barX_pi[j + 1] = barX_pi[j] + DT * (barD1_pi[j] + barD2_pi_j);
    }
}

extern "C" {

// Returns pointer to JSON string (owned by g_output, valid until next call)
EMSCRIPTEN_KEEPALIVE
const char* solve_single(double p1, double p2, double b1, double b2, double r1, double r2) {
    g_output.clear();
    g_output.reserve(64 * 1024);  // ~64KB typical for single solve
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
    std::array<double, N> V1_arr, V2_arr;
    for (int j = 0; j < N; ++j) {
        double V1 = 0.0, V2 = 0.0;
        for (int z = 0; z <= j; ++z) {
            V1 += eq.env.Xtilde2[j][z].dot(bba1.barHk[j][z]);
            V2 += eq.env.Xtilde1[j][z].dot(bba2.barHk[j][z]);
        }
        V1_arr[j] = V1 * p2 * p2 * DT;
        V2_arr[j] = V2 * p1 * p1 * DT;
    }

    std::array<double, N> barD1_pi, barX_pi;
    compute_perfect_info(b1, b2, barD1_pi, barX_pi);

    const auto& tg = t_grid();
    out("{");
    out_array("t", tg.data(), N);
    out(",\"residuals\":[");
    for (size_t i = 0; i < eq.residuals.size(); ++i)
        out("%s%.6g", i ? "," : "", eq.residuals[i]);
    out("],\"n_iters\":%zu", eq.residuals.size());

    out(","); out_kernel2d("X", eq.env.X);
    out(","); out_kernel2d("D1", eq.D1);
    out(","); out_kernel2d("D2", eq.D2);
    out(","); out_kernel2d("calD1", eq.calD1);

    out(","); out_array("barD1", bar.barD1.data(), N);
    out(","); out_array("barD2", bar.barD2.data(), N);
    out(","); out_array("barX", bar.barX.data(), N);
    out(",\"bar_residual\":%.6g", bar.bar_residual);
    out(","); out_array("V1", V1_arr.data(), N);
    out(","); out_array("V2", V2_arr.data(), N);
    out(","); out_array("barD1_pi", barD1_pi.data(), N);
    out(","); out_array("barX_pi", barX_pi.data(), N);
    out(",\"J1\":%.6g,\"J2\":%.6g", costs.J1, costs.J2);
    out(",\"p1\":%.6g,\"p2\":%.6g,\"b1\":%.6g,\"b2\":%.6g", p1, p2, b1, b2);
    out("}");

    return g_output.c_str();
}

EMSCRIPTEN_KEEPALIVE
const char* solve_sweep(double p1, double b1, double b2, double r1, double r2,
                         const double* p2_vals, int n_p2) {
    g_output.clear();
    g_output.reserve(32 * 1024);  // ~32KB typical for sweep
    g_b1 = b1; g_b2 = b2;
    g_r1 = r1; g_r2 = r2;

    std::array<double, N> barD1_pi, barX_pi;
    compute_perfect_info(b1, b2, barD1_pi, barX_pi);

    const auto& tg = t_grid();
    out("{");
    out_array("t", tg.data(), N);
    out(","); out_array("barD1_pi", barD1_pi.data(), N);
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
        out(","); out_array("barD1", bar.barD1.data(), N);
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
        std::array<double, N> V1_arr, V2_arr;
        for (int j = 0; j < N; ++j) {
            double V1 = 0.0, V2 = 0.0;
            for (int z = 0; z <= j; ++z) {
                V1 += eq.env.Xtilde2[j][z].dot(bba1.barHk[j][z]);
                V2 += eq.env.Xtilde1[j][z].dot(bba2.barHk[j][z]);
            }
            V1_arr[j] = V1 * p2 * p2 * DT;
            V2_arr[j] = V2 * p1 * p1 * DT;
        }
        out(","); out_array("V1", V1_arr.data(), N);
        out(","); out_array("V2", V2_arr.data(), N);
        out("}");
    }
    out("]}");

    return g_output.c_str();
}

} // extern "C"
