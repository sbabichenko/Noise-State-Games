// Test the S/R costate decomposition: Hx = S·X + R
//
// Verifies:
//  1. The decomposition is numerically exact (S·X + R ≡ Hx by construction)
//  2. The standard and SR backward passes produce identical Hx
//  3. Reports decomposition quality metrics (||R||/||Hx||, ||V_R||/||V||)
//  4. Compares S with the perfect-information Riccati S_pi

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

int main() {
    struct TestCase { int N; double p1, p2, r1, r2; const char* desc; };
    TestCase cases[] = {
        {40, 3.0, 3.0, RHO, RHO,   "symmetric baseline"},
        {40, 5.0, 2.0, RHO, RHO,   "asymmetric precision"},
        {40, 3.0, 3.0, 0.05, 0.2,  "asymmetric costs"},
        {40, 5.0, 2.0, 0.05, 0.2,  "asymmetric p + r"},
        {60, 3.0, 3.0, RHO, RHO,   "N=60 symmetric"},
        {40, 8.0, 8.0, RHO, RHO,   "high precision"},
        {40, 1.0, 1.0, RHO, RHO,   "low precision"},
    };

    std::printf("=== S/R Costate Decomposition Test ===\n\n");

    bool all_pass = true;

    for (auto& tc : cases) {
        SolverContext ctx{};
        ctx.n = tc.N;
        ctx.T = 1.0;
        ctx.b1 = B1_DEFAULT;
        ctx.b2 = B2_DEFAULT;
        ctx.r1 = tc.r1;
        ctx.r2 = tc.r2;
        ctx.sigma = 1.0;
        ctx.x0 = 0.0;
        ScopedSolverContext guard(ctx);

        std::printf("--- %s (N=%d, p=(%.1f,%.1f), r=(%.3f,%.3f)) ---\n",
                    tc.desc, tc.N, tc.p1, tc.p2, g_r1, g_r2);

        // 1. Solve equilibrium (standard)
        auto t0 = Clock::now();
        auto eq = solve_equilibrium(tc.p1, tc.p2, false);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int iters = static_cast<int>(eq.residuals.size());
        bool conv = !eq.residuals.empty() && eq.residuals.back() < PICARD_TOL;
        std::printf("  Solve: %.1f ms, %d iters, %s (resid=%.2e)\n",
                    ms, iters, conv ? "converged" : "NOT CONVERGED",
                    eq.residuals.empty() ? -1.0 : eq.residuals.back());

        auto prec1 = make_constant_prec(tc.p1 * tc.p1);
        auto prec2 = make_constant_prec(tc.p2 * tc.p2);

        // 2. Standard backward pass (Hx only)
        Kernel2D Hx1_std, Hx2_std;
        backward_kernels(eq.env.X, eq.env.Xtilde2, eq.D2, prec2, 0.0, Hx1_std);
        backward_kernels(eq.env.X, eq.env.Xtilde1, eq.D1, prec1, 0.0, Hx2_std);

        // 3. SR backward pass
        Kernel2D Hx1_sr, Hx2_sr, V1, V2;
        CostateDecomposition d1, d2;
        backward_kernels_sr(eq.env.X, eq.env.Xtilde2, eq.D2, prec2, 0.0, Hx1_sr, V1, d1);
        backward_kernels_sr(eq.env.X, eq.env.Xtilde1, eq.D1, prec1, 0.0, Hx2_sr, V2, d2);

        // 4. Verify Hx_std ≡ Hx_sr (must be identical)
        int tri = g_n * (g_n + 1) / 2;
        double hx_diff1 = 0.0, hx_diff2 = 0.0, hx_norm1 = 0.0, hx_norm2 = 0.0;
        for (int i = 0; i < tri; ++i) {
            hx_diff1 += (Hx1_std.data[i] - Hx1_sr.data[i]).squaredNorm();
            hx_diff2 += (Hx2_std.data[i] - Hx2_sr.data[i]).squaredNorm();
            hx_norm1 += Hx1_std.data[i].squaredNorm();
            hx_norm2 += Hx2_std.data[i].squaredNorm();
        }
        double gap1 = std::sqrt(hx_diff1) / std::max(1e-15, std::sqrt(hx_norm1));
        double gap2 = std::sqrt(hx_diff2) / std::max(1e-15, std::sqrt(hx_norm2));
        bool hx_match = (gap1 < 1e-12 && gap2 < 1e-12);
        std::printf("  Hx match (std vs SR): player1=%.2e, player2=%.2e  %s\n",
                    gap1, gap2, hx_match ? "OK" : "FAIL");
        if (!hx_match) all_pass = false;

        // 5. Verify decomposition: S·X + R == Hx (exact by construction)
        double max_err1 = 0.0, max_err2 = 0.0;
        for (int j = 0; j < g_n; ++j)
            for (int r = 0; r <= j; ++r) {
                Vec3 check1 = d1.S[j] * eq.env.X[j][r] + d1.R[j][r] - Hx1_sr[j][r];
                Vec3 check2 = d2.S[j] * eq.env.X[j][r] + d2.R[j][r] - Hx2_sr[j][r];
                max_err1 = std::max(max_err1, check1.norm());
                max_err2 = std::max(max_err2, check2.norm());
            }
        bool decomp_exact = (max_err1 < 1e-12 && max_err2 < 1e-12);
        std::printf("  Decomposition error: player1=%.2e, player2=%.2e  %s\n",
                    max_err1, max_err2, decomp_exact ? "OK" : "FAIL");
        if (!decomp_exact) all_pass = false;

        // 6. Report decomposition quality
        std::printf("  Player 1: ||R||/||Hx|| = %.4f, ||V_R||/||V|| = %.4f\n",
                    d1.R_frac, d1.V_R_frac);
        std::printf("  Player 2: ||R||/||Hx|| = %.4f, ||V_R||/||V|| = %.4f\n",
                    d2.R_frac, d2.V_R_frac);

        // 7. Compare S with perfect-info Riccati
        std::array<double, N_MAX> S_pi;
        compute_perfect_info_S(S_pi);

        double s1_gap_sq = 0.0, s2_gap_sq = 0.0, spi_sq = 0.0;
        for (int j = 0; j < g_n; ++j) {
            s1_gap_sq += (d1.S[j] - S_pi[j]) * (d1.S[j] - S_pi[j]);
            s2_gap_sq += (d2.S[j] - S_pi[j]) * (d2.S[j] - S_pi[j]);
            spi_sq += S_pi[j] * S_pi[j];
        }
        double s1_rel = std::sqrt(s1_gap_sq) / std::max(1e-15, std::sqrt(spi_sq));
        double s2_rel = std::sqrt(s2_gap_sq) / std::max(1e-15, std::sqrt(spi_sq));
        std::printf("  ||S1 - S_pi||/||S_pi|| = %.6f\n", s1_rel);
        std::printf("  ||S2 - S_pi||/||S_pi|| = %.6f\n", s2_rel);

        // 8. Print S at sample points
        int n4 = g_n / 4, n2 = g_n / 2, n34 = 3 * g_n / 4;
        std::printf("  S values at t = 0, T/4, T/2, 3T/4, T:\n");
        std::printf("    S1:  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f\n",
                    d1.S[0], d1.S[n4], d1.S[n2], d1.S[n34], d1.S[g_n - 1]);
        std::printf("    S2:  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f\n",
                    d2.S[0], d2.S[n4], d2.S[n2], d2.S[n34], d2.S[g_n - 1]);
        std::printf("    Spi: %8.5f  %8.5f  %8.5f  %8.5f  %8.5f\n",
                    S_pi[0], S_pi[n4], S_pi[n2], S_pi[n34], S_pi[g_n - 1]);

        // 9. Print V_S (the wedge's Riccati-forcing term)
        std::printf("  V_S (wedge scalar) at t = 0, T/4, T/2, 3T/4, T:\n");
        std::printf("    V_S1: %9.6f  %9.6f  %9.6f  %9.6f  %9.6f\n",
                    d1.V_S[0], d1.V_S[n4], d1.V_S[n2], d1.V_S[n34], d1.V_S[g_n - 1]);
        std::printf("    V_S2: %9.6f  %9.6f  %9.6f  %9.6f  %9.6f\n",
                    d2.V_S[0], d2.V_S[n4], d2.V_S[n2], d2.V_S[n34], d2.V_S[g_n - 1]);

        std::printf("\n");
    }

    std::printf("=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
