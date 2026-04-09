// Test convergence at large T with large r (expensive control) and small p (poor obs).
// Hypothesis: large r dampens the Picard map, enabling convergence at larger T.

#include "lqg_solver.h"
#include <cstdio>
#include <cmath>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

static double kernel_gap(const Kernel2D& a1, const Kernel2D& a2,
                          const Kernel2D& b1, const Kernel2D& b2) {
    int tri = g_n * (g_n + 1) / 2;
    double diff = 0.0, norm = 0.0;
    for (int i = 0; i < tri; ++i) {
        diff += (a1.data[i] - b1.data[i]).squaredNorm()
              + (a2.data[i] - b2.data[i]).squaredNorm();
        norm += a1.data[i].squaredNorm() + a2.data[i].squaredNorm();
    }
    return std::sqrt(diff) / std::max(1e-15, std::sqrt(norm));
}

int main() {
    int N = 40;

    // Part 1: Sweep r from small to large at various T
    std::printf("=== Part 1: Convergence vs r at various T (p=3, symmetric) ===\n\n");
    double r_values[] = {0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0};
    double T_values[] = {1.0, 2.0, 3.0, 5.0, 10.0, 20.0};

    std::printf("  %8s", "r\\T");
    for (double T : T_values)
        std::printf("  %10.0f", T);
    std::printf("\n");

    for (double r : r_values) {
        std::printf("  %8.2f", r);
        for (double T : T_values) {
            SolverContext ctx{};
            ctx.n = N; ctx.T = T;
            ctx.b1 = B1_DEFAULT; ctx.b2 = B2_DEFAULT;
            ctx.r1 = r; ctx.r2 = r;
            ctx.sigma = 1.0; ctx.x0 = 0.0;
            ScopedSolverContext guard(ctx);

            auto eq = solve_equilibrium(3.0, 3.0, false);
            int iters = (int)eq.residuals.size();
            bool conv = !eq.residuals.empty() &&
                        std::isfinite(eq.residuals.back()) &&
                        eq.residuals.back() < PICARD_TOL;

            char buf[20];
            if (conv) std::snprintf(buf, sizeof(buf), "%d", iters);
            else std::snprintf(buf, sizeof(buf), "%dF", iters);
            std::printf("  %10s", buf);
        }
        std::printf("\n");
    }

    // Part 2: Sweep p from large to small at various T
    std::printf("\n=== Part 2: Convergence vs p at various T (r=0.1, symmetric) ===\n\n");
    double p_values[] = {0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0};

    std::printf("  %8s", "p\\T");
    for (double T : T_values)
        std::printf("  %10.0f", T);
    std::printf("\n");

    for (double p : p_values) {
        std::printf("  %8.1f", p);
        for (double T : T_values) {
            SolverContext ctx{};
            ctx.n = N; ctx.T = T;
            ctx.b1 = B1_DEFAULT; ctx.b2 = B2_DEFAULT;
            ctx.r1 = RHO; ctx.r2 = RHO;
            ctx.sigma = 1.0; ctx.x0 = 0.0;
            ScopedSolverContext guard(ctx);

            auto eq = solve_equilibrium(p, p, false);
            int iters = (int)eq.residuals.size();
            bool conv = !eq.residuals.empty() &&
                        std::isfinite(eq.residuals.back()) &&
                        eq.residuals.back() < PICARD_TOL;

            char buf[20];
            if (conv) std::snprintf(buf, sizeof(buf), "%d", iters);
            else std::snprintf(buf, sizeof(buf), "%dF", iters);
            std::printf("  %10s", buf);
        }
        std::printf("\n");
    }

    // Part 3: Large r + large T uniqueness check
    std::printf("\n=== Part 3: Uniqueness at large r, large T ===\n\n");
    struct UniqueCase { double T, p, r; const char* desc; };
    UniqueCase ucases[] = {
        {5.0,  3.0, 1.0,  "T=5  r=1"},
        {5.0,  3.0, 5.0,  "T=5  r=5"},
        {5.0,  3.0, 10.0, "T=5  r=10"},
        {10.0, 3.0, 5.0,  "T=10 r=5"},
        {10.0, 3.0, 10.0, "T=10 r=10"},
        {10.0, 3.0, 50.0, "T=10 r=50"},
        {20.0, 3.0, 10.0, "T=20 r=10"},
        {20.0, 3.0, 50.0, "T=20 r=50"},
        {20.0, 3.0, 100.0,"T=20 r=100"},
        {5.0,  0.5, 1.0,  "T=5  p=0.5 r=1"},
        {10.0, 0.5, 5.0,  "T=10 p=0.5 r=5"},
        {20.0, 0.5, 10.0, "T=20 p=0.5 r=10"},
    };

    for (auto& uc : ucases) {
        SolverContext ctx{};
        ctx.n = N; ctx.T = uc.T;
        ctx.b1 = B1_DEFAULT; ctx.b2 = B2_DEFAULT;
        ctx.r1 = uc.r; ctx.r2 = uc.r;
        ctx.sigma = 1.0; ctx.x0 = 0.0;
        ScopedSolverContext guard(ctx);

        auto eq_cold = solve_equilibrium(uc.p, uc.p, false);
        bool conv_cold = !eq_cold.residuals.empty() &&
                         std::isfinite(eq_cold.residuals.back()) &&
                         eq_cold.residuals.back() < PICARD_TOL;

        auto eq_fast = solve_equilibrium_fast(uc.p, uc.p, false);
        bool conv_fast = !eq_fast.residuals.empty() &&
                         std::isfinite(eq_fast.residuals.back()) &&
                         eq_fast.residuals.back() < PICARD_TOL;

        // Negative warm start
        std::array<double, N_MAX> S_pi;
        S_pi.fill(0.0);
        S_pi[g_n - 1] = TERMINAL_STATE_WEIGHT;
        for (int j = g_n - 2; j >= 0; --j)
            S_pi[j] = S_pi[j + 1]
                + g_dt * (1.0 - (2.0 / uc.r) * S_pi[j + 1] * S_pi[j + 1]);
        Vec3 sigE0 = g_sigma * E0();
        Kernel2D D1_neg, D2_neg;
        for (int j = 0; j < g_n; ++j)
            for (int s = 0; s <= j; ++s) {
                D1_neg[j][s] = (S_pi[j] / uc.r) * sigE0;  // opposite sign
                D2_neg[j][s] = (S_pi[j] / uc.r) * sigE0;
            }
        auto eq_neg = solve_equilibrium_warm(uc.p, uc.p, D1_neg, D2_neg, false);
        bool conv_neg = !eq_neg.residuals.empty() &&
                        std::isfinite(eq_neg.residuals.back()) &&
                        eq_neg.residuals.back() < PICARD_TOL;

        double gap_fast = (conv_cold && conv_fast) ?
            kernel_gap(eq_cold.D1, eq_cold.D2, eq_fast.D1, eq_fast.D2) : -1;
        double gap_neg = (conv_cold && conv_neg) ?
            kernel_gap(eq_cold.D1, eq_cold.D2, eq_neg.D1, eq_neg.D2) : -1;

        bool all_conv = conv_cold && conv_fast && conv_neg;
        bool unique = all_conv && gap_fast < 1e-3 && gap_neg < 1e-3;

        std::printf("  %-22s  cold=%4d%s  fast=%4d%s  neg=%4d%s",
                    uc.desc,
                    (int)eq_cold.residuals.size(), conv_cold ? "" : "F",
                    (int)eq_fast.residuals.size(), conv_fast ? "" : "F",
                    (int)eq_neg.residuals.size(), conv_neg ? "" : "F");
        if (all_conv)
            std::printf("  gap=(%.1e,%.1e)  %s\n", gap_fast, gap_neg,
                        unique ? "UNIQUE" : "DIFFERENT!");
        else
            std::printf("  INCOMPLETE\n");
    }

    return 0;
}
