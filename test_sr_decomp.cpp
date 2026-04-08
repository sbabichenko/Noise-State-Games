// Test the S/R costate decomposition and SR-initialized solver performance.
//
// Part 1: Verify decomposition correctness (S·X + R ≡ Hx)
// Part 2: Benchmark solve_equilibrium vs solve_equilibrium_sr
//         — iteration count, wall time, and solution agreement

#include "lqg_solver.h"
#include <chrono>
#include <cstdio>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

static void compute_perfect_info_S(std::array<double, N_MAX>& S_pi) {
    S_pi.fill(0.0);
    S_pi[g_n - 1] = TERMINAL_STATE_WEIGHT;
    for (int j = g_n - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1] + g_dt * (1.0 - (1.0 / g_r1 + 1.0 / g_r2) * S_pi[j + 1] * S_pi[j + 1]);
}

static double kernel_gap(const Kernel2D& a1, const Kernel2D& a2,
                          const Kernel2D& b1, const Kernel2D& b2) {
    int tri = g_n * (g_n + 1) / 2;
    double diff = 0.0, norm = 0.0;
    for (int i = 0; i < tri; ++i) {
        diff += (a1.data[i] - b1.data[i]).squaredNorm();
        diff += (a2.data[i] - b2.data[i]).squaredNorm();
        norm += a1.data[i].squaredNorm() + a2.data[i].squaredNorm();
    }
    return std::sqrt(diff) / std::max(1e-15, std::sqrt(norm));
}

int main() {
    struct TestCase { int N; double p1, p2, r1, r2; const char* desc; };
    TestCase cases[] = {
        {40,  3.0, 3.0, RHO,  RHO,  "symmetric baseline"},
        {40,  5.0, 2.0, RHO,  RHO,  "asymmetric precision"},
        {40,  3.0, 3.0, 0.05, 0.2,  "asymmetric costs"},
        {40,  5.0, 2.0, 0.05, 0.2,  "asymmetric p + r"},
        {60,  3.0, 3.0, RHO,  RHO,  "N=60 symmetric"},
        {80,  3.0, 3.0, RHO,  RHO,  "N=80 symmetric"},
        {40,  8.0, 8.0, RHO,  RHO,  "high precision"},
        {40,  1.0, 1.0, RHO,  RHO,  "low precision"},
        {40, 10.0, 1.0, 0.05, 0.3,  "extreme asymmetry"},
    };

    bool all_pass = true;

    // ========== Part 1: Decomposition correctness ==========
    std::printf("========== Part 1: S/R Decomposition Correctness ==========\n\n");

    for (auto& tc : cases) {
        SolverContext ctx{};
        ctx.n = tc.N; ctx.T = 1.0;
        ctx.b1 = B1_DEFAULT; ctx.b2 = B2_DEFAULT;
        ctx.r1 = tc.r1; ctx.r2 = tc.r2;
        ctx.sigma = 1.0; ctx.x0 = 0.0;
        ScopedSolverContext guard(ctx);

        auto eq = solve_equilibrium(tc.p1, tc.p2, false);
        auto prec1 = make_constant_prec(tc.p1 * tc.p1);
        auto prec2 = make_constant_prec(tc.p2 * tc.p2);

        // Standard backward (Hx only)
        Kernel2D Hx1_std, Hx2_std;
        backward_kernels(eq.env.X, eq.env.Xtilde2, eq.D2, prec2, 0.0, Hx1_std);
        backward_kernels(eq.env.X, eq.env.Xtilde1, eq.D1, prec1, 0.0, Hx2_std);

        // SR backward
        Kernel2D Hx1_sr, Hx2_sr, V1, V2;
        CostateDecomposition d1, d2;
        backward_kernels_sr(eq.env.X, eq.env.Xtilde2, eq.D2, prec2, 0.0, Hx1_sr, V1, d1);
        backward_kernels_sr(eq.env.X, eq.env.Xtilde1, eq.D1, prec1, 0.0, Hx2_sr, V2, d2);

        // Verify Hx identical
        double gap = kernel_gap(Hx1_std, Hx2_std, Hx1_sr, Hx2_sr);
        bool match = gap < 1e-12;
        if (!match) all_pass = false;

        // Verify S·X + R == Hx
        double max_err = 0.0;
        for (int j = 0; j < g_n; ++j)
            for (int r = 0; r <= j; ++r) {
                max_err = std::max(max_err,
                    (d1.S[j] * eq.env.X[j][r] + d1.R[j][r] - Hx1_sr[j][r]).norm());
                max_err = std::max(max_err,
                    (d2.S[j] * eq.env.X[j][r] + d2.R[j][r] - Hx2_sr[j][r]).norm());
            }
        bool exact = max_err < 1e-12;
        if (!exact) all_pass = false;

        std::array<double, N_MAX> S_pi;
        compute_perfect_info_S(S_pi);
        double s1_gap = 0.0, s2_gap = 0.0, spi_sq = 0.0;
        for (int j = 0; j < g_n; ++j) {
            s1_gap += (d1.S[j] - S_pi[j]) * (d1.S[j] - S_pi[j]);
            s2_gap += (d2.S[j] - S_pi[j]) * (d2.S[j] - S_pi[j]);
            spi_sq += S_pi[j] * S_pi[j];
        }

        std::printf("  %-22s  Hx=%s  S·X+R=%s  ||R||/||Hx||=(%.3f,%.3f)  ||S-Spi||=(%.3f,%.3f)\n",
                    tc.desc,
                    match ? "OK" : "FAIL",
                    exact ? "OK" : "FAIL",
                    d1.R_frac, d2.R_frac,
                    std::sqrt(s1_gap / std::max(1e-15, spi_sq)),
                    std::sqrt(s2_gap / std::max(1e-15, spi_sq)));
    }

    // ========== Part 2: Performance benchmark ==========
    std::printf("\n========== Part 2: Performance Benchmark ==========\n\n");
    std::printf("%-24s  %6s  %6s  %6s  %8s  %7s  %7s  %7s  %7s  %8s  %8s  %s\n",
                "test", "std_it", "sr_it", "fst_it",
                "std_ms", "sr_ms", "fst_ms",
                "sr_spd", "fst_spd", "sr_gap", "fst_gap", "status");
    std::printf("%-24s  %6s  %6s  %6s  %8s  %7s  %7s  %7s  %7s  %8s  %8s  %s\n",
                "----", "------", "-----", "------",
                "------", "------", "------",
                "------", "------", "------", "------", "------");

    for (auto& tc : cases) {
        SolverContext ctx{};
        ctx.n = tc.N; ctx.T = 1.0;
        ctx.b1 = B1_DEFAULT; ctx.b2 = B2_DEFAULT;
        ctx.r1 = tc.r1; ctx.r2 = tc.r2;
        ctx.sigma = 1.0; ctx.x0 = 0.0;
        ScopedSolverContext guard(ctx);

        // Standard solver (average of 3 runs)
        double ms_std_total = 0.0;
        int it_std = 0;
        EquilibriumResult eq_std;
        for (int run = 0; run < 3; ++run) {
            auto t0 = Clock::now();
            eq_std = solve_equilibrium(tc.p1, tc.p2, false);
            auto t1 = Clock::now();
            ms_std_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            it_std = static_cast<int>(eq_std.residuals.size());
        }
        double ms_std = ms_std_total / 3.0;

        // SR solver (average of 3 runs)
        double ms_sr_total = 0.0;
        int it_sr = 0;
        EquilibriumResult eq_sr;
        for (int run = 0; run < 3; ++run) {
            auto t0 = Clock::now();
            eq_sr = solve_equilibrium_sr(tc.p1, tc.p2, false);
            auto t1 = Clock::now();
            ms_sr_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            it_sr = static_cast<int>(eq_sr.residuals.size());
        }
        double ms_sr = ms_sr_total / 3.0;

        // Fast solver (average of 3 runs)
        double ms_fst_total = 0.0;
        int it_fst = 0;
        EquilibriumResult eq_fst;
        for (int run = 0; run < 3; ++run) {
            auto t0 = Clock::now();
            eq_fst = solve_equilibrium_fast(tc.p1, tc.p2, false);
            auto t1 = Clock::now();
            ms_fst_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            it_fst = static_cast<int>(eq_fst.residuals.size());
        }
        double ms_fst = ms_fst_total / 3.0;

        // Compare solutions
        double sr_gap = kernel_gap(eq_std.D1, eq_std.D2, eq_sr.D1, eq_sr.D2);
        double fst_gap = kernel_gap(eq_std.D1, eq_std.D2, eq_fst.D1, eq_fst.D2);

        bool std_conv = !eq_std.residuals.empty() && eq_std.residuals.back() < PICARD_TOL;
        bool sr_conv  = !eq_sr.residuals.empty()  && eq_sr.residuals.back()  < PICARD_TOL;
        bool fst_conv = !eq_fst.residuals.empty() && eq_fst.residuals.back() < PICARD_TOL;
        bool sr_close  = sr_gap  < 0.05;
        bool fst_close = fst_gap < 0.15;  // fast solver trades accuracy for speed
        if (!sr_close || !fst_close) all_pass = false;

        double sr_speedup  = ms_std / std::max(0.01, ms_sr);
        double fst_speedup = ms_std / std::max(0.01, ms_fst);
        std::printf("%-24s  %6d  %6d  %6d  %7.1fms  %6.1fms  %6.1fms  %6.2fx  %6.2fx  %8.5f  %8.5f  %s/%s/%s%s%s\n",
                    tc.desc, it_std, it_sr, it_fst,
                    ms_std, ms_sr, ms_fst,
                    sr_speedup, fst_speedup,
                    sr_gap, fst_gap,
                    std_conv ? "y" : "N", sr_conv ? "y" : "N", fst_conv ? "y" : "N",
                    sr_close ? "" : " SR_GAP!", fst_close ? "" : " FST_GAP!");
    }

    std::printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
