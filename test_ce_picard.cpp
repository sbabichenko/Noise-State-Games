// Compare standard Picard iteration vs CE-based Picard iteration
//
// Checks: convergence, iteration count, and agreement of solutions.

#include "lqg_solver.h"
#include <chrono>
#include <cstdio>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

int main() {
    std::printf("%-4s  %-8s  %-8s  %-6s  %-8s  %-6s  %-8s  %-12s\n",
                "N", "gain", "std_ms", "std_it", "ce_ms", "ce_it", "D_gap", "converged");

    for (auto [N, g1, g2] : std::vector<std::tuple<int,double,double>>{
            {20, 3.0, 3.0},
            {40, 3.0, 3.0},
            {40, 5.0, 2.0},
            {60, 3.0, 3.0},
            {80, 3.0, 3.0}}) {

        set_grid(N, 1.0);
        g_b1 = B1_DEFAULT; g_b2 = B2_DEFAULT;
        g_r1 = RHO; g_r2 = RHO; g_sigma = 1.0;

        // Standard solver
        auto t0 = Clock::now();
        auto eq_std = solve_equilibrium(g1, g2, false);
        auto t1 = Clock::now();
        double ms_std = std::chrono::duration<double,std::milli>(t1-t0).count();
        int it_std = static_cast<int>(eq_std.residuals.size());

        // CE-based solver
        auto t2 = Clock::now();
        auto eq_ce = solve_equilibrium_ce(g1, g2, false);
        auto t3 = Clock::now();
        double ms_ce = std::chrono::duration<double,std::milli>(t3-t2).count();
        int it_ce = static_cast<int>(eq_ce.residuals.size());

        // Compare D kernels
        double d_diff = 0.0, d_norm = 0.0;
        int tri = g_n * (g_n + 1) / 2;
        for (int i = 0; i < tri; ++i) {
            d_diff += (eq_std.D1.data[i] - eq_ce.D1.data[i]).squaredNorm();
            d_diff += (eq_std.D2.data[i] - eq_ce.D2.data[i]).squaredNorm();
            d_norm += eq_std.D1.data[i].squaredNorm() + eq_std.D2.data[i].squaredNorm();
        }
        double gap = std::sqrt(d_diff) / std::max(1e-15, std::sqrt(d_norm));

        bool std_conv = !eq_std.residuals.empty() && eq_std.residuals.back() < PICARD_TOL;
        bool ce_conv = !eq_ce.residuals.empty() && eq_ce.residuals.back() < PICARD_TOL;

        std::printf("%-4d  (%.0f,%.0f)  %7.1f  %6d  %7.1f  %6d  %8.5f  %s/%s\n",
                    N, g1, g2, ms_std, it_std, ms_ce, it_ce, gap,
                    std_conv ? "yes" : "NO", ce_conv ? "yes" : "NO");
    }
}
