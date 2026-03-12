// CLI tool for the interactive Dash app.
//
// Modes:
//   solve_interactive single <p1> <p2> <b1> <b2>
//     → JSON with kernels, bar solution, wedges, costs for one (p1, p2)
//
//   solve_interactive sweep <p1> <b1> <b2> <p2_0> <p2_1> ... <p2_n>
//     → JSON array: for each p2, private and pooled costs + barD1 curves

#include "lqg_solver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_array(const char* name, const double* arr, int n) {
    printf("\"%s\":[", name);
    for (int i = 0; i < n; ++i)
        printf("%s%.12g", i ? "," : "", arr[i]);
    printf("]");
}

static void print_kernel2d(const char* name, const Kernel2D& K) {
    printf("\"%s\":{", name);
    bool first = true;
    for (int ch = 0; ch < 3; ++ch) {
        printf("%s\"ch%d\":[", first ? "" : ",", ch);
        first = false;
        bool first_row = true;
        for (int ti = 0; ti < g_n; ++ti) {
            printf("%s[", first_row ? "" : ",");
            first_row = false;
            for (int si = 0; si <= ti; ++si)
                printf("%s%.12g", si ? "," : "", K[ti][si](ch));
            for (int si = ti + 1; si < g_n; ++si)
                printf(",0");
            printf("]");
        }
        printf("]");
    }
    printf("}");
}

// Perfect-info Riccati solution (depends on b1, b2 for the bar dynamics)
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

static int run_single(int argc, char* argv[]) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s single <p1> <p2> <b1> <b2> <r1> <r2>\n", argv[0]);
        return 1;
    }
    double p1 = atof(argv[2]), p2 = atof(argv[3]);
    double b1 = atof(argv[4]), b2 = atof(argv[5]);
    double r1 = atof(argv[6]), r2 = atof(argv[7]);
    g_b1 = b1; g_b2 = b2;
    g_r1 = r1; g_r2 = r2;

    auto eq = solve_equilibrium(p1, p2, false);
    auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                      p1*p1, p2*p2, 2000, 0.08, 1e-10);
    auto costs = compute_costs_general(eq.env, eq.calD1, eq.calD2, bar, r1, r2, b1, b2);

    // Wedges
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
    printf("{");
    print_array("t", tg.data(), g_n);

    printf(",\"residuals\":[");
    for (size_t i = 0; i < eq.residuals.size(); ++i)
        printf("%s%.12g", i ? "," : "", eq.residuals[i]);
    printf("],\"n_iters\":%zu", eq.residuals.size());

    printf(","); print_kernel2d("X", eq.env.X);
    printf(","); print_kernel2d("D1", eq.D1);
    printf(","); print_kernel2d("D2", eq.D2);
    printf(","); print_kernel2d("calD1", eq.calD1);
    printf(","); print_kernel2d("calD2", eq.calD2);
    printf(","); print_kernel2d("Xtilde1", eq.env.Xtilde1);
    printf(","); print_kernel2d("Xtilde2", eq.env.Xtilde2);

    printf(","); print_array("barD1", bar.barD1.data(), g_n);
    printf(","); print_array("barD2", bar.barD2.data(), g_n);
    printf(","); print_array("barX", bar.barX.data(), g_n);
    printf(",\"bar_residual\":%.12g", bar.bar_residual);

    printf(","); print_array("V1", V1_arr.data(), g_n);
    printf(","); print_array("V2", V2_arr.data(), g_n);

    printf(","); print_array("barD1_pi", barD1_pi.data(), g_n);
    printf(","); print_array("barX_pi", barX_pi.data(), g_n);

    printf(",\"J1\":%.12g,\"J2\":%.12g", costs.J1, costs.J2);
    printf(",\"p1\":%.12g,\"p2\":%.12g,\"b1\":%.12g,\"b2\":%.12g", p1, p2, b1, b2);
    printf("}\n");
    return 0;
}

static int run_sweep(int argc, char* argv[]) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s sweep <p1> <b1> <b2> <r1> <r2> <p2_0> [p2_1 ...]\n", argv[0]);
        return 1;
    }
    double p1 = atof(argv[2]);
    double b1 = atof(argv[3]), b2 = atof(argv[4]);
    double r1 = atof(argv[5]), r2 = atof(argv[6]);
    g_b1 = b1; g_b2 = b2;
    g_r1 = r1; g_r2 = r2;

    int n_p2 = argc - 7;
    std::vector<double> p2_vals(n_p2);
    for (int i = 0; i < n_p2; ++i)
        p2_vals[i] = atof(argv[7 + i]);

    std::array<double, N_MAX> barD1_pi, barX_pi;
    compute_perfect_info(b1, b2, barD1_pi, barX_pi);

    const auto& tg = t_grid();
    printf("{");
    print_array("t", tg.data(), g_n);
    printf(","); print_array("barD1_pi", barD1_pi.data(), g_n);
    printf(",\"p1\":%.12g,\"b1\":%.12g,\"b2\":%.12g", p1, b1, b2);

    printf(",\"sweeps\":[");
    for (int i = 0; i < n_p2; ++i) {
        double p2 = p2_vals[i];
        if (i > 0) printf(",");
        printf("{\"p2\":%.12g", p2);

        // Private equilibrium
        auto eq = solve_equilibrium(p1, p2, false);
        auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                          p1*p1, p2*p2, 2000, 0.08, 1e-10);
        auto costs_priv = compute_costs_general(eq.env, eq.calD1, eq.calD2,
                                                 bar, r1, r2, b1, b2);

        printf(","); print_array("barD1", bar.barD1.data(), g_n);
        printf(","); print_array("barD2", bar.barD2.data(), g_n);
        printf(","); print_array("barX", bar.barX.data(), g_n);
        printf(",\"J1_priv\":%.12g,\"J2_priv\":%.12g", costs_priv.J1, costs_priv.J2);
        printf(",\"n_iters\":%zu", eq.residuals.size());

        // Pooled equilibrium: both players see same signal through Pi1
        double p_common = std::sqrt(p1*p1 + p2*p2);
        auto eq_pool = solve_equilibrium(p_common, p_common, false,
                                          Pi1(), 1, Pi1(), 1);
        auto bar_pool = solve_bar_equilibrium(eq_pool.env, eq_pool.D1, eq_pool.D2,
                                               p_common*p_common, p_common*p_common,
                                               2000, 0.08, 1e-10);
        auto costs_pool = compute_costs_general(eq_pool.env, eq_pool.calD1, eq_pool.calD2,
                                                 bar_pool, r1, r2, b1, b2);

        printf(",\"J1_pool\":%.12g,\"J2_pool\":%.12g", costs_pool.J1, costs_pool.J2);
        printf(",\"p_common\":%.12g", p_common);

        // Wedges (private)
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
        printf(","); print_array("V1", V1_arr.data(), g_n);
        printf(","); print_array("V2", V2_arr.data(), g_n);

        printf("}");
    }
    printf("]}\n");
    return 0;
}

int main(int argc, char* argv[]) {
    // Initialize grid to default (N=40, T=1.0) — matches previous compile-time constants
    set_grid(40, 1.0);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <single|sweep> ...\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "sweep") == 0)
        return run_sweep(argc, argv);
    if (strcmp(argv[1], "single") == 0)
        return run_single(argc, argv);

    // Legacy: positional args without subcommand
    if (argc >= 5) {
        // Shift args to match single mode
        char* new_argv[] = {argv[0], (char*)"single", argv[1], argv[2], argv[3], argv[4]};
        return run_single(6, new_argv);
    }
    fprintf(stderr, "Usage: %s <single|sweep> ...\n", argv[0]);
    return 1;
}
