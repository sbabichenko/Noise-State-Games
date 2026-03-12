// CLI tool for the interactive Dash app.
// Usage: solve_interactive <p1> <p2> <b1> <b2>
// Outputs JSON with all kernel/control/cost data for one parameter set.

#include "lqg_solver.h"
#include <cstdio>
#include <cstdlib>

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
        for (int ti = 0; ti < N; ++ti) {
            printf("%s[", first_row ? "" : ",");
            first_row = false;
            for (int si = 0; si <= ti; ++si)
                printf("%s%.12g", si ? "," : "", K[ti][si](ch));
            // pad with zeros for si > ti (upper triangle)
            for (int si = ti + 1; si < N; ++si)
                printf(",0");
            printf("]");
        }
        printf("]");
    }
    printf("}");
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <p1> <p2> <b1> <b2>\n", argv[0]);
        return 1;
    }

    double p1 = atof(argv[1]);
    double p2 = atof(argv[2]);
    double b1 = atof(argv[3]);
    double b2 = atof(argv[4]);

    g_b1 = b1;
    g_b2 = b2;

    auto eq = solve_equilibrium(p1, p2, false);

    auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                      p1 * p1, p2 * p2, 2000, 0.08, 1e-10);

    auto costs = compute_costs_general(eq.env, eq.calD1, eq.calD2,
                                        bar, RHO, b1, b2);

    // Backward bar adjoints for wedges
    auto prec1_arr = make_constant_prec(p1 * p1);
    auto prec2_arr = make_constant_prec(p2 * p2);
    auto bba1 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde2, eq.D2,
                                       bar.barX, b1, prec2_arr, 0.0);
    auto bba2 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde1, eq.D1,
                                       bar.barX, b2, prec1_arr, 0.0);

    // Compute wedges V1, V2
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

    // Perfect info benchmark
    std::array<double, N> S_pi;
    S_pi.fill(0.0);
    S_pi[N - 1] = TERMINAL_STATE_WEIGHT;
    for (int j = N - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1] + DT * (1.0 - (2.0 / RHO) * S_pi[j + 1] * S_pi[j + 1]);

    std::array<double, N> barD1_pi, barX_pi;
    barX_pi[0] = X0;
    for (int j = 0; j < N; ++j) {
        barD1_pi[j] = -(1.0 / RHO) * S_pi[j] * (barX_pi[j] - b1);
        double barD2_pi_j = -(1.0 / RHO) * S_pi[j] * (barX_pi[j] - b2);
        if (j < N - 1)
            barX_pi[j + 1] = barX_pi[j] + DT * (barD1_pi[j] + barD2_pi_j);
    }

    // Output JSON
    const auto& tg = t_grid();
    printf("{");

    // t_grid
    print_array("t", tg.data(), N);

    // Residuals
    printf(",\"residuals\":[");
    for (size_t i = 0; i < eq.residuals.size(); ++i)
        printf("%s%.12g", i ? "," : "", eq.residuals[i]);
    printf("],\"n_iters\":%zu", eq.residuals.size());

    // Kernels
    printf(","); print_kernel2d("X", eq.env.X);
    printf(","); print_kernel2d("D1", eq.D1);
    printf(","); print_kernel2d("D2", eq.D2);
    printf(","); print_kernel2d("calD1", eq.calD1);
    printf(","); print_kernel2d("calD2", eq.calD2);
    printf(","); print_kernel2d("Xtilde1", eq.env.Xtilde1);
    printf(","); print_kernel2d("Xtilde2", eq.env.Xtilde2);

    // Bar solution
    printf(","); print_array("barD1", bar.barD1.data(), N);
    printf(","); print_array("barD2", bar.barD2.data(), N);
    printf(","); print_array("barX", bar.barX.data(), N);
    printf(",\"bar_residual\":%.12g", bar.bar_residual);

    // Wedges
    printf(","); print_array("V1", V1_arr.data(), N);
    printf(","); print_array("V2", V2_arr.data(), N);

    // Perfect info
    printf(","); print_array("barD1_pi", barD1_pi.data(), N);
    printf(","); print_array("barX_pi", barX_pi.data(), N);

    // Costs
    printf(",\"J1\":%.12g,\"J2\":%.12g", costs.J1, costs.J2);

    // Parameters echo
    printf(",\"p1\":%.12g,\"p2\":%.12g,\"b1\":%.12g,\"b2\":%.12g", p1, p2, b1, b2);

    printf("}\n");
    return 0;
}
