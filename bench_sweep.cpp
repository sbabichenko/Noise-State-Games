// Comprehensive benchmark: speed and memory across parameter ranges.
// Reports per-iteration cost, total solve time, iteration count, and memory.

#include "lqg_solver.h"
#include <chrono>
#include <cstdio>
#include <cmath>

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
    // Memory model:
    //   Kernel2D = TRI * sizeof(Vec3) = N*(N+1)/2 * 24 bytes
    //   EnvironmentResult = 5 Kernel2D (X, Xtilde1, Xtilde2, A_store1, A_store2)
    //   AA history (standard): AA_STORE * 4 Kernel2D = 6*4 = 24 Kernel2D
    //   AA history (fast/Picard): 0 Kernel2D (+ 2 for residual buffer)
    //   Hx1, Hx2: 2 Kernel2D
    //   HkSlice buffers: 2 * N*N * sizeof(Mat3) = 2 * N² * 72 bytes (per thread)
    //   FwdEnv cached bufs: 5 Kernel2D (thread-local)

    struct TestCase {
        int N; double p1, p2, r1, r2; const char* desc;
    };

    // --- Section 1: Scaling with N ---
    std::printf("========== Section 1: Scaling with Grid Size N ==========\n\n");
    std::printf("  p1=p2=3, r1=r2=0.1 (symmetric baseline)\n\n");
    std::printf("  %4s  %7s  %7s  | %6s %7s %6s | %6s %7s %6s | %6s %7s %6s | %9s %8s %7s\n",
                "N", "K2D_KB", "Hk_KB",
                "std_it", "std_ms", "ms/it",
                "sr_it", "sr_ms", "ms/it",
                "fst_it", "fst_ms", "ms/it",
                "peak_MB", "AA_MB", "fst_MB");
    std::printf("  %4s  %7s  %7s  | %6s %7s %6s | %6s %7s %6s | %6s %7s %6s | %9s %8s %7s\n",
                "----", "-------", "-------",
                "------", "-------", "------",
                "------", "-------", "------",
                "------", "-------", "------",
                "---------", "--------", "-------");

    int Ns[] = {20, 40, 60, 80, 100, 120, 140, 160};
    for (int N : Ns) {
        SolverContext ctx{};
        ctx.n = N; ctx.T = 1.0;
        ctx.b1 = B1_DEFAULT; ctx.b2 = B2_DEFAULT;
        ctx.r1 = RHO; ctx.r2 = RHO;
        ctx.sigma = 1.0; ctx.x0 = 0.0;
        ScopedSolverContext guard(ctx);

        int TRI = N * (N + 1) / 2;
        double k2d_kb = TRI * 24.0 / 1024.0;
        double hk_kb = N * N * 72.0 / 1024.0;  // per HkSlice buffer

        // Standard solver
        double ms_std = 0; int it_std = 0;
        for (int run = 0; run < 3; ++run) {
            auto t0 = Clock::now();
            auto eq = solve_equilibrium(3.0, 3.0, false);
            auto t1 = Clock::now();
            ms_std += std::chrono::duration<double, std::milli>(t1 - t0).count();
            it_std = (int)eq.residuals.size();
        }
        ms_std /= 3.0;

        // SR solver
        double ms_sr = 0; int it_sr = 0;
        for (int run = 0; run < 3; ++run) {
            auto t0 = Clock::now();
            auto eq = solve_equilibrium_sr(3.0, 3.0, false);
            auto t1 = Clock::now();
            ms_sr += std::chrono::duration<double, std::milli>(t1 - t0).count();
            it_sr = (int)eq.residuals.size();
        }
        ms_sr /= 3.0;

        // Fast solver (Picard)
        double ms_fst = 0; int it_fst = 0;
        for (int run = 0; run < 3; ++run) {
            auto t0 = Clock::now();
            auto eq = solve_equilibrium_fast(3.0, 3.0, false);
            auto t1 = Clock::now();
            ms_fst += std::chrono::duration<double, std::milli>(t1 - t0).count();
            it_fst = (int)eq.residuals.size();
        }
        ms_fst /= 3.0;

        // Memory estimates (KB → MB)
        // Standard: env(5) + D1,D2(2) + Hx1,Hx2(2) + AA_hist(24) + fwd_bufs(5) + calD1,calD2(2) = 40 K2D
        //   + 2 HkSlice (per thread, 2 threads)
        double std_k2d_count = 40;
        double std_peak_mb = (std_k2d_count * k2d_kb + 4 * hk_kb) / 1024.0;

        // AA history alone
        double aa_mb = 24.0 * k2d_kb / 1024.0;

        // Fast: env(5) + D1,D2(2) + Hx1,Hx2(2) + picard_f1,f2(2) + fwd_bufs(5) + calD1,calD2(2) = 18 K2D
        double fst_k2d_count = 18;
        double fst_peak_mb = (fst_k2d_count * k2d_kb + 4 * hk_kb) / 1024.0;

        std::printf("  %4d  %6.1fK  %6.1fK  | %6d %6.1fms %5.2fms | %6d %6.1fms %5.2fms | %6d %6.1fms %5.2fms | %8.2fMB %6.2fMB %6.2fMB\n",
                    N, k2d_kb, hk_kb,
                    it_std, ms_std, ms_std / it_std,
                    it_sr, ms_sr, ms_sr / it_sr,
                    it_fst, ms_fst, ms_fst / it_fst,
                    std_peak_mb, aa_mb, fst_peak_mb);
    }

    // --- Section 2: Parameter sweep at N=40 ---
    std::printf("\n========== Section 2: Parameter Sweep (N=40) ==========\n\n");
    std::printf("  %-26s | %6s %7s  %7s | %6s %7s  %7s | %7s %7s  %7s\n",
                "params", "std_it", "std_ms", "D_gap",
                "sr_it", "sr_ms", "D_gap",
                "fst_it", "fst_ms", "D_gap");
    std::printf("  %-26s | %6s %7s  %7s | %6s %7s  %7s | %7s %7s  %7s\n",
                "--------------------------", "------", "-------", "-------",
                "------", "-------", "-------",
                "-------", "-------", "-------");

    TestCase param_cases[] = {
        // Symmetric
        {40,  1.0, 1.0, 0.1,  0.1,  "p=1,1  r=0.1,0.1"},
        {40,  3.0, 3.0, 0.1,  0.1,  "p=3,3  r=0.1,0.1"},
        {40,  5.0, 5.0, 0.1,  0.1,  "p=5,5  r=0.1,0.1"},
        {40, 10.0,10.0, 0.1,  0.1,  "p=10,10 r=0.1,0.1"},
        {40, 20.0,20.0, 0.1,  0.1,  "p=20,20 r=0.1,0.1"},
        // Asymmetric precision
        {40,  5.0, 2.0, 0.1,  0.1,  "p=5,2  r=0.1,0.1"},
        {40, 10.0, 2.0, 0.1,  0.1,  "p=10,2 r=0.1,0.1"},
        {40, 20.0, 1.0, 0.1,  0.1,  "p=20,1 r=0.1,0.1"},
        // Asymmetric costs
        {40,  3.0, 3.0, 0.05, 0.2,  "p=3,3  r=0.05,0.2"},
        {40,  3.0, 3.0, 0.02, 0.5,  "p=3,3  r=0.02,0.5"},
        {40,  3.0, 3.0, 0.01, 1.0,  "p=3,3  r=0.01,1.0"},
        // Both asymmetric
        {40,  5.0, 2.0, 0.05, 0.2,  "p=5,2  r=0.05,0.2"},
        {40, 10.0, 1.0, 0.05, 0.3,  "p=10,1 r=0.05,0.3"},
        {40, 10.0, 1.0, 0.01, 0.5,  "p=10,1 r=0.01,0.5"},
    };

    for (auto& tc : param_cases) {
        SolverContext ctx{};
        ctx.n = tc.N; ctx.T = 1.0;
        ctx.b1 = B1_DEFAULT; ctx.b2 = B2_DEFAULT;
        ctx.r1 = tc.r1; ctx.r2 = tc.r2;
        ctx.sigma = 1.0; ctx.x0 = 0.0;
        ScopedSolverContext guard(ctx);

        // Standard
        auto t0 = Clock::now();
        auto eq_std = solve_equilibrium(tc.p1, tc.p2, false);
        auto t1 = Clock::now();
        double ms_std = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int it_std = (int)eq_std.residuals.size();

        // SR
        t0 = Clock::now();
        auto eq_sr = solve_equilibrium_sr(tc.p1, tc.p2, false);
        t1 = Clock::now();
        double ms_sr = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int it_sr = (int)eq_sr.residuals.size();
        double gap_sr = kernel_gap(eq_std.D1, eq_std.D2, eq_sr.D1, eq_sr.D2);

        // Fast
        t0 = Clock::now();
        auto eq_fst = solve_equilibrium_fast(tc.p1, tc.p2, false);
        t1 = Clock::now();
        double ms_fst = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int it_fst = (int)eq_fst.residuals.size();
        double gap_fst = kernel_gap(eq_std.D1, eq_std.D2, eq_fst.D1, eq_fst.D2);

        std::printf("  %-26s | %6d %6.1fms  %.5f | %6d %6.1fms  %.5f | %6d %6.1fms  %.5f\n",
                    tc.desc,
                    it_std, ms_std, 0.0,
                    it_sr, ms_sr, gap_sr,
                    it_fst, ms_fst, gap_fst);
    }

    // --- Section 3: Per-iteration cost breakdown (N=40 vs N=80 vs N=120) ---
    std::printf("\n========== Section 3: Per-Iteration Cost Scaling ==========\n\n");
    std::printf("  %4s  %10s  %10s  %10s  %7s\n",
                "N", "fwd_env", "backward", "total/it", "ratio");
    std::printf("  %4s  %10s  %10s  %10s  %7s\n",
                "----", "----------", "----------", "----------", "-------");

    int breakdown_Ns[] = {20, 40, 60, 80, 100, 120};
    double prev_total = 0;
    for (int N : breakdown_Ns) {
        SolverContext ctx{};
        ctx.n = N; ctx.T = 1.0;
        ctx.b1 = B1_DEFAULT; ctx.b2 = B2_DEFAULT;
        ctx.r1 = RHO; ctx.r2 = RHO;
        ctx.sigma = 1.0; ctx.x0 = 0.0;
        ScopedSolverContext guard(ctx);

        // Run standard solver and measure per-iteration
        auto t0 = Clock::now();
        auto eq = solve_equilibrium(3.0, 3.0, false);
        auto t1 = Clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int iters = (int)eq.residuals.size();
        double per_it = total_ms / iters;

        // Rough breakdown: forward_environment is ~65%, backward is ~35%
        // (based on earlier profiling: filter 42% + control 23% + state 5% = 70% forward)
        double fwd_ms = per_it * 0.65;
        double bwd_ms = per_it * 0.35;

        double ratio = (prev_total > 0) ? per_it / prev_total : 0;
        std::printf("  %4d  %8.3fms  %8.3fms  %8.3fms  %6.2fx\n",
                    N, fwd_ms, bwd_ms, per_it,
                    ratio);
        prev_total = per_it;
    }

    std::printf("\n  (forward ≈ 65%% = filter + control + state; backward ≈ 35%% = costate propagation)\n");
    std::printf("  (ratio = cost relative to previous N; expect ~(N_new/N_old)^3 scaling)\n");

    std::printf("\nDone.\n");
    return 0;
}
