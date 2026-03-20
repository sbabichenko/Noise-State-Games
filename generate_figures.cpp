// ============================================================
// Generate all paper figures for the decentralized LQG game.
// Outputs CSV data files to data/ directory; plot with plot_figures.py
// ============================================================

#include "lqg_solver.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

static const std::string DATA_DIR = "data";

// ---------- equilibrium cache ----------
// Many figures solve the same (p1, p2, obs_config) equilibria.
// Cache them to avoid redundant ~0.02s solves.
// All entries are pre-computed in parallel before figure generation.
static std::map<std::tuple<int,int,int,int>, EquilibriumResult> eq_cache;

static auto make_eq_key(double p1, double p2, int obs1, int obs2) {
    return std::make_tuple(
        static_cast<int>(std::llround(p1 * 1000000)), static_cast<int>(std::llround(p2 * 1000000)),
        obs1, obs2);
}

static EquilibriumResult& cached_solve(
    double p1, double p2, bool verbose = false,
    const Mat3& Pi_1 = Pi1(), int obs1 = 1,
    const Mat3& Pi_2 = Pi2(), int obs2 = 2) {
    auto key = make_eq_key(p1, p2, obs1, obs2);
    auto it = eq_cache.find(key);
    if (it != eq_cache.end()) return it->second;
    auto [ins, _] = eq_cache.emplace(key,
        solve_equilibrium(p1, p2, verbose, Pi_1, obs1, Pi_2, obs2));
    return ins->second;
}

// ---------- bar solution cache ----------
// Key: (eq_key, prec1*1M, prec2*1M, b1*1M, b2*1M)
using BarKey = std::tuple<int,int,int,int,int,int,int,int>;
static std::map<BarKey, BarSolution> bar_cache;

static BarSolution& cached_bar_solve(
    const EquilibriumResult& eq, double prec1, double prec2,
    double p1, double p2, int obs1, int obs2,
    int max_iters = 2000, double relax = 0.08, double tol = 1e-10) {
    auto key = std::make_tuple(
        static_cast<int>(std::llround(p1 * 1000000)), static_cast<int>(std::llround(p2 * 1000000)),
        obs1, obs2,
        static_cast<int>(std::llround(prec1 * 1000000)), static_cast<int>(std::llround(prec2 * 1000000)),
        static_cast<int>(std::llround(g_b1 * 1000000)), static_cast<int>(std::llround(g_b2 * 1000000)));
    auto it = bar_cache.find(key);
    if (it != bar_cache.end()) return it->second;
    auto [ins, _] = bar_cache.emplace(key,
        solve_bar_equilibrium(eq.env, eq.D1, eq.D2, prec1, prec2, max_iters, relax, tol));
    return ins->second;
}

// ---------- CSV helpers ----------
// Write a 2D kernel (g_n x g_n x D_W) as CSV: columns t, s, ch0, ch1, ch2
// Only writes entries where s <= t (causal kernel)
static void write_kernel2d(const std::string& path, const Kernel2D& K) {
    std::ofstream f(path);
    f << std::setprecision(12);
    f << "t_idx,s_idx,t,s,ch0,ch1,ch2\n";
    const auto& tg = t_grid();
    for (int ti = 0; ti < g_n; ++ti)
        for (int si = 0; si <= ti; ++si)
            f << ti << "," << si << "," << tg[ti] << "," << tg[si] << ","
              << K[ti][si](0) << "," << K[ti][si](1) << "," << K[ti][si](2) << "\n";
}

// Write F1[T] kernel: F1[g_n-1][u][s](row, col) for all u, s, row, col
static void write_F1_T(const std::string& path, const Kernel3D& F1) {
    std::ofstream f(path);
    f << std::setprecision(12);
    f << "u_idx,s_idx,u,s,row,col,value\n";
    const auto& tg = t_grid();
    for (int u = 0; u < g_n; ++u)
        for (int s = 0; s < g_n; ++s)
            for (int row = 0; row < D_W; ++row)
                for (int col = 0; col < D_W; ++col)
                    f << u << "," << s << "," << tg[u] << "," << tg[s] << ","
                      << row << "," << col << "," << F1[g_n-1][u][s](row, col) << "\n";
}

// ---------- main ----------
int main() {
    // Initialize grid to default (N=40, T=1.0)
    set_grid(40, 1.0);

    fs::create_directories(DATA_DIR);
    const auto& tg = t_grid();

    std::vector<int> p_values = {1, 2, 3, 5, 10};
    int p1_fixed = 3;
    std::vector<int> p2_values = {1, 2, 3, 5, 10, 20};

    // ============================================================
    // Pre-solve all unique equilibria in parallel
    // ============================================================
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Pre-solving all equilibria in parallel ...\n";

    // Collect all unique solve parameters
    struct SolveSpec {
        double p1, p2;
        Mat3 Pi_1, Pi_2;
        int obs1, obs2;
    };
    std::vector<SolveSpec> specs;
    std::map<std::tuple<int,int,int,int>, int> spec_dedup;

    auto add_spec = [&](double p1, double p2, const Mat3& Pi_1, int obs1,
                        const Mat3& Pi_2, int obs2) {
        auto key = make_eq_key(p1, p2, obs1, obs2);
        if (spec_dedup.count(key) == 0) {
            spec_dedup[key] = static_cast<int>(specs.size());
            specs.push_back({p1, p2, Pi_1, Pi_2, obs1, obs2});
        }
    };

    // (3,3) benchmark
    add_spec(3, 3, Pi1(), 1, Pi2(), 2);
    // Fig 8: symmetric p=1,2,3,5,10
    for (int p : p_values)
        add_spec(p, p, Pi1(), 1, Pi2(), 2);
    // Fig 10/11: asymmetric p1=3, p2 varies
    for (int p2v : p2_values)
        add_spec(p1_fixed, p2v, Pi1(), 1, Pi2(), 2);
    // Fig 12: pooled info
    for (int p2v : p_values) {
        double p_common = std::sqrt(p1_fixed * p1_fixed + p2v * p2v);
        add_spec(p_common, p_common, Pi1(), 1, Pi1(), 1);
    }

    std::cout << "  " << specs.size() << " unique equilibria to solve\n";

    // Solve (3,3) first with verbose output + timing
    auto t0 = std::chrono::high_resolution_clock::now();
    cached_solve(3, 3, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Solve (3,3) time: "
              << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    // Solve remaining in parallel (exclude (3,3) which is already done)
    auto t_par0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < specs.size(); ++i) {
        auto& sp = specs[i];
        auto key = make_eq_key(sp.p1, sp.p2, sp.obs1, sp.obs2);
        // Skip if already in cache (e.g. the (3,3) we just solved)
        bool need_solve;
        #pragma omp critical
        { need_solve = (eq_cache.find(key) == eq_cache.end()); }
        if (!need_solve) continue;

        auto result = solve_equilibrium(sp.p1, sp.p2, false,
                                         sp.Pi_1, sp.obs1, sp.Pi_2, sp.obs2);
        #pragma omp critical
        { eq_cache.emplace(key, std::move(result)); }
    }
    auto t_par1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Parallel solve time: "
              << std::chrono::duration<double>(t_par1 - t_par0).count() << "s\n";

    auto& eq = cached_solve(3, 3);
    auto& D1 = eq.D1;
    auto& D2 = eq.D2;
    auto& env = eq.env;

    auto bar_sol = solve_bar_equilibrium(env, D1, D2, 9, 9, 500, 0.35, 1e-10);
    std::cout << "  Bar residual: " << bar_sol.bar_residual << "\n";

    // Backward bar adjoints for Figure 9
    auto prec9 = make_constant_prec(9.0);
    auto bba = backward_bar_adjoints(env.X, env.Xtilde2, D2, bar_sol.barX, g_b1, prec9, 0.0);

    // ============================================================
    // FIGURE 3: Picard residual
    // ============================================================
    std::cout << "Figure 3: Picard residual ...\n";
    {
        std::ofstream f(DATA_DIR + "/fig3_residuals.csv");
        f << std::setprecision(12);
        f << "iteration,residual\n";
        for (size_t i = 0; i < eq.residuals.size(); ++i)
            f << i << "," << eq.residuals[i] << "\n";
    }

    // ============================================================
    // FIGURE 4: State kernel X(t,s)
    // ============================================================
    std::cout << "Figure 4: State kernel X(t,s) ...\n";
    write_kernel2d(DATA_DIR + "/fig4_X.csv", env.X);

    // ============================================================
    // FIGURE 5: D1(t,s) feedback kernel
    // ============================================================
    std::cout << "Figure 5: D1(t,s) feedback kernel ...\n";
    write_kernel2d(DATA_DIR + "/fig5_D1.csv", D1);

    // ============================================================
    // FIGURE 6: calD1(t,s) primitive-noise kernel
    // ============================================================
    std::cout << "Figure 6: calD1(t,s) primitive-noise kernel ...\n";
    write_kernel2d(DATA_DIR + "/fig6_calD1.csv", eq.calD1);

    // ============================================================
    // FIGURE 7: F1 at t=T
    // ============================================================
    std::cout << "Figure 7: Filtering kernel F1 at t=T ...\n";
    {
        Kernel3D F1;
        materialize_F(env.Xtilde1, env.A_store1, env.obs_gain1, env.obs_idx1, F1);
        write_F1_T(DATA_DIR + "/fig7_F1_T.csv", F1);
    }

    // ============================================================
    // FIGURE 8: Mean control barD1(t) vs precision p
    // ============================================================
    std::cout << "Figure 8: Mean control vs precision ...\n";

    // Perfect-info benchmark
    std::array<double, N_MAX> S_pi;
    S_pi.fill(0.0);
    S_pi[g_n - 1] = TERMINAL_STATE_WEIGHT;
    for (int j = g_n - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1] + g_dt * (1.0 - (2.0 / RHO) * S_pi[j + 1] * S_pi[j + 1]);

    std::array<double, N_MAX> barD1_pi, barX_pi;
    barD1_pi.fill(0.0);
    barX_pi.fill(0.0);
    barX_pi[0] = X0;
    for (int j = 0; j < g_n; ++j) {
        barD1_pi[j] = -(1.0 / RHO) * S_pi[j] * (barX_pi[j] - B1_DEFAULT);
        double barD2_pi_j = -(1.0 / RHO) * S_pi[j] * (barX_pi[j] - B2_DEFAULT);
        if (j < g_n - 1)
            barX_pi[j + 1] = barX_pi[j] + g_dt * (barD1_pi[j] + barD2_pi_j);
    }

    {
        std::ofstream f(DATA_DIR + "/fig8_barD1.csv");
        f << std::setprecision(12);
        f << "t,perfect_info";
        for (int p : p_values)
            f << ",p" << p;
        f << "\n";

        std::map<int, std::array<double, N_MAX>> barD1_curves;

        for (int p : p_values) {
            std::cout << "  Solving p=" << p << " ...\n";
            auto& eq_p = cached_solve(p, p);
            auto& bar_p = cached_bar_solve(eq_p, p*p, p*p, p, p, 1, 2);
            barD1_curves[p] = bar_p.barD1;
            std::cout << "    bar residual: " << bar_p.bar_residual << "\n";
        }

        for (int i = 0; i < g_n; ++i) {
            f << tg[i] << "," << barD1_pi[i];
            for (int p : p_values)
                f << "," << barD1_curves[p][i];
            f << "\n";
        }
    }

    // ============================================================
    // FIGURE 9: barH2 and Xtilde2
    // ============================================================
    std::cout << "Figure 9: Opponent adjoint barH2 and gain Xtilde2 ...\n";
    write_kernel2d(DATA_DIR + "/fig9_barH2.csv", bba.barHk);
    write_kernel2d(DATA_DIR + "/fig9_R2.csv", env.Xtilde2);

    // ============================================================
    // FIGURE 10: Asymmetric equilibrium panels (p1=3, p2 varies)
    // ============================================================
    std::cout << "Figure 10: Asymmetric equilibrium panels (p1=3, p2 varies) ...\n";

    {
        std::ofstream f(DATA_DIR + "/fig10_asymmetric.csv");
        f << std::setprecision(12);
        f << "t,p2,barD1,barD2,barX\n";

        for (int p2v : p2_values) {
            std::cout << "  Solving p1=" << p1_fixed << ", p2=" << p2v << " ...\n";
            auto& eq_a = cached_solve(p1_fixed, p2v);
            auto& bar_a = cached_bar_solve(eq_a, p1_fixed*p1_fixed, p2v*p2v,
                                           p1_fixed, p2v, 1, 2);
            std::cout << "    bar residual: " << bar_a.bar_residual << "\n";

            for (int i = 0; i < g_n; ++i)
                f << tg[i] << "," << p2v << ","
                  << bar_a.barD1[i] << "," << bar_a.barD2[i] << ","
                  << bar_a.barX[i] << "\n";
        }

        // Also write perfect-info effort benchmark
        std::ofstream fpi(DATA_DIR + "/fig10_perfect_info.csv");
        fpi << std::setprecision(12);
        fpi << "t,barD1_pi\n";
        for (int i = 0; i < g_n; ++i)
            fpi << tg[i] << "," << barD1_pi[i] << "\n";
    }

    // ============================================================
    // FIGURE 11: Information wedges V^1(t) and V^2(t) as p2 varies
    // ============================================================
    std::cout << "Figure 11: Information wedges V^1(t) and V^2(t) ...\n";
    {
        std::ofstream f(DATA_DIR + "/fig11_wedges.csv");
        f << std::setprecision(12);
        f << "t,p2,V1,V2\n";

        for (int p2v : p2_values) {
            std::cout << "  Computing wedges for p1=" << p1_fixed << ", p2=" << p2v << " ...\n";
            auto& eq_a = cached_solve(p1_fixed, p2v);
            auto& bar_a = cached_bar_solve(eq_a, p1_fixed*p1_fixed, p2v*p2v,
                                           p1_fixed, p2v, 1, 2);

            auto prec2_arr = make_constant_prec(static_cast<double>(p2v * p2v));
            auto prec1_arr = make_constant_prec(static_cast<double>(p1_fixed * p1_fixed));

            auto bba1 = backward_bar_adjoints(eq_a.env.X, eq_a.env.Xtilde2, eq_a.D2,
                                              bar_a.barX, B1_DEFAULT, prec2_arr, 0.0);
            auto bba2 = backward_bar_adjoints(eq_a.env.X, eq_a.env.Xtilde1, eq_a.D1,
                                              bar_a.barX, B2_DEFAULT, prec1_arr, 0.0);

            for (int j = 0; j < g_n; ++j) {
                int m = j + 1;
                double V1 = 0.0, V2 = 0.0;
                for (int z = 0; z < m; ++z) {
                    V1 += eq_a.env.Xtilde2[j][z].dot(bba1.barHk[j][z]);
                    V2 += eq_a.env.Xtilde1[j][z].dot(bba2.barHk[j][z]);
                }
                V1 *= static_cast<double>(p2v * p2v) * g_dt;
                V2 *= static_cast<double>(p1_fixed * p1_fixed) * g_dt;

                f << tg[j] << "," << p2v << "," << V1 << "," << V2 << "\n";
            }

            std::cout << "    done\n";
        }
    }

    // ============================================================
    // FIGURE 12: Player costs — private vs pooled
    // ============================================================
    std::cout << "Figure 12: Player costs, private vs pooled ...\n";

    struct TargetConfig {
        double b1, b2;
        std::string label;
    };
    std::vector<TargetConfig> configs = {
        {1.0, -1.0, "competitive"},
        {0.0,  0.0, "cooperative"},
    };

    {
        std::ofstream f(DATA_DIR + "/fig12_costs.csv");
        f << std::setprecision(12);
        f << "config,p2,J1_priv,J2_priv,J1_pool,J2_pool\n";

        for (auto& cfg : configs) {
            std::cout << "\n  === targets=(" << cfg.b1 << "," << cfg.b2 << ") ===\n";
            g_b1 = cfg.b1;
            g_b2 = cfg.b2;

            for (int p2v : p_values) {
                // Private
                std::cout << "    Private: p2=" << p2v << " ...\n";
                auto& eq_a = cached_solve(p1_fixed, p2v);
                auto& bar_a = cached_bar_solve(eq_a, p1_fixed*p1_fixed, p2v*p2v,
                                               p1_fixed, p2v, 1, 2);
                auto [j1p, j2p_priv] = compute_costs_general(eq_a.env, eq_a.calD1, eq_a.calD2, bar_a, g_r1, g_r2, cfg.b1, cfg.b2);

                // Pooled
                double p_common = std::sqrt(p1_fixed * p1_fixed + p2v * p2v);
                std::cout << "    Pooled: p_common=" << p_common << " ...\n";
                auto& eq_c = cached_solve(p_common, p_common, false,
                                          Pi1(), 1, Pi1(), 1);
                auto& bar_c = cached_bar_solve(eq_c, p_common*p_common, p_common*p_common,
                                               p_common, p_common, 1, 1);
                auto [j1pool, j2pool] = compute_costs_general(eq_c.env, eq_c.calD1, eq_c.calD2, bar_c, g_r1, g_r2, cfg.b1, cfg.b2);

                f << cfg.label << "," << p2v << ","
                  << j1p << "," << j2p_priv << ","
                  << j1pool << "," << j2pool << "\n";
            }

            g_b1 = B1_DEFAULT;
            g_b2 = B2_DEFAULT;
        }
    }

    // ============================================================
    // FIGURE 13: Precision allocation sweep
    // Budget: p1^2 + p2^2 = Pbar (total actual precision)
    // Symmetric costs r1=r2=0.1, sigma=0.5, T=1
    // Competitive (theta1=1, theta2=-1) and cooperative (theta1=theta2=0)
    // ============================================================
    std::cout << "\nFigure 13: Precision allocation sweep ...\n";

    // Clear caches since we're changing parameters
    eq_cache.clear();
    bar_cache.clear();

    const double PBAR = 20.0;       // total actual precision budget: p1^2 + p2^2 = PBAR
    const double R1_ALLOC = 0.1;    // symmetric control costs
    const double R2_ALLOC = 0.1;
    const double SIGMA_ALLOC = 0.5; // reduced state diffusion amplifies strategic effects
    const int N_SWEEP = 61;         // number of sweep points

    g_r1 = R1_ALLOC;
    g_r2 = R2_ALLOC;
    g_sigma = SIGMA_ALLOC;

    struct AllocTarget {
        double b1, b2;
        std::string label;
    };

    std::vector<AllocTarget> alloc_configs = {
        {1.0, -1.0, "competitive"},
        {0.0,  0.0, "cooperative"},
    };

    // Budget: p1^2 + p2^2 = PBAR  =>  p2 = sqrt(PBAR - p1^2)
    // Sweep p1 (root precision) from near 0 to near sqrt(PBAR)
    const double P1_MAX_ROOT = std::sqrt(PBAR);  // max root precision for player 1
    const double MARGIN = 0.15;
    std::vector<double> p1_sweep(N_SWEEP);
    for (int i = 0; i < N_SWEEP; ++i)
        p1_sweep[i] = MARGIN + (P1_MAX_ROOT - 2.0 * MARGIN) * i / (N_SWEEP - 1);

    // Precompute p2 values from budget constraint
    std::vector<double> p2_sweep(N_SWEEP);
    for (int i = 0; i < N_SWEEP; ++i)
        p2_sweep[i] = std::sqrt(PBAR - p1_sweep[i] * p1_sweep[i]);

    std::cout << "  Pre-solving " << N_SWEEP << " equilibria for precision sweep (warm-started) ...\n";
    {
        // Solve from the middle outward for best warm-start quality
        int mid = N_SWEEP / 2;
        std::vector<EquilibriumResult> eq_results(N_SWEEP);

        // Cold-start from the middle
        eq_results[mid] = solve_equilibrium(p1_sweep[mid], p2_sweep[mid], false);

        // Sweep rightward from mid
        for (int i = mid + 1; i < N_SWEEP; ++i) {
            eq_results[i] = solve_equilibrium_warm(
                p1_sweep[i], p2_sweep[i],
                eq_results[i - 1].D1, eq_results[i - 1].D2, false);
        }
        // Sweep leftward from mid
        for (int i = mid - 1; i >= 0; --i) {
            eq_results[i] = solve_equilibrium_warm(
                p1_sweep[i], p2_sweep[i],
                eq_results[i + 1].D1, eq_results[i + 1].D2, false);
        }

        // Insert into cache
        for (int i = 0; i < N_SWEEP; ++i) {
            auto key = make_eq_key(p1_sweep[i], p2_sweep[i], 1, 2);
            eq_cache.emplace(key, std::move(eq_results[i]));
        }
    }
    std::cout << "  Equilibria solved.\n";

    // Full-info benchmark: large precision so both players effectively observe X
    // Sweep over multiple (r1, r2) configurations
    struct RConfig {
        double r1, r2;
        std::string label;
    };
    std::vector<RConfig> r_configs = {
        {0.05, 0.05, "r0.05_0.05"},
        {0.1, 0.1, "r0.1_0.1"},
        {0.05, 0.2, "r0.05_0.2"},
        {0.2, 0.2, "r0.2_0.2"},
        {0.1, 0.5, "r0.1_0.5"},
        {0.5, 0.5, "r0.5_0.5"},
    };

    {
        std::ofstream f(DATA_DIR + "/fig13_precision_allocation.csv");
        f << std::setprecision(12);
        f << "config,r_config,r1,r2,p1_root,p1_prec,J1_eq,J2_eq,Jtotal_eq,J1_fi,J2_fi,Jtotal_fi,V1_total,V2_total,barD1sq_eq,barD2sq_eq,barD1sq_fi,barD2sq_fi\n";

        for (auto& rcfg : r_configs) {
            g_r1 = rcfg.r1;
            g_r2 = rcfg.r2;

            // Clear caches for new r values
            eq_cache.clear();
            bar_cache.clear();

            // Full-info benchmark: coupled Riccati for two-player Nash with full observation
            // Ṡ_i = S_i²/r_i + 2·S_i·S_j/r_j − 1,  S_i(T)=0
            // Variance cost for player i: ∫ σ² S_i dt
            std::array<double, N_MAX> S_fi1{}, S_fi2{};
            S_fi1[g_n - 1] = TERMINAL_STATE_WEIGHT;
            S_fi2[g_n - 1] = TERMINAL_STATE_WEIGHT;
            for (int j = g_n - 2; j >= 0; --j) {
                double s1 = S_fi1[j + 1], s2 = S_fi2[j + 1];
                S_fi1[j] = s1 + g_dt * (1.0 - s1 * s1 / rcfg.r1 - 2.0 * s1 * s2 / rcfg.r2);
                S_fi2[j] = s2 + g_dt * (1.0 - s2 * s2 / rcfg.r2 - 2.0 * s1 * s2 / rcfg.r1);
            }
            // Variance costs
            double J1_fi_var = 0.0, J2_fi_var = 0.0;
            for (int j = 0; j < g_n; ++j) {
                J1_fi_var += SIGMA_ALLOC * SIGMA_ALLOC * S_fi1[j] * g_dt;
                J2_fi_var += SIGMA_ALLOC * SIGMA_ALLOC * S_fi2[j] * g_dt;
            }
            // Full-info control costs from variance: ∫ (S_i/r_i)² σ² · Var(X) dt
            // We need the full-info Var(X)(t).  dVar/dt = -2(S1/r1 + S2/r2)*Var + σ²
            std::array<double, N_MAX> Var_fi{};
            Var_fi[0] = 0.0;  // X₀ deterministic
            for (int j = 0; j < g_n - 1; ++j) {
                double k_sum = S_fi1[j] / rcfg.r1 + S_fi2[j] / rcfg.r2;
                Var_fi[j + 1] = Var_fi[j] + g_dt * (-2.0 * k_sum * Var_fi[j] + SIGMA_ALLOC * SIGMA_ALLOC);
            }
            // Full variance cost: J_i_var = ∫ (1 + r_i K_i²) Var dt  where K_i = S_i/r_i
            // But from the value function: J_i_var = S_i(0)·Var(0) + ∫σ²S_i dt = ∫σ²S_i dt  (since Var(0)=0)
            // So J1_fi_var and J2_fi_var are already correct.

            std::cout << "\n  Full-info Riccati for " << rcfg.label
                      << ": J1_var=" << J1_fi_var << " J2_var=" << J2_fi_var << "\n";

            // Pre-solve equilibria for this r config (warm-started from middle)
            std::cout << "\n  Pre-solving for " << rcfg.label << " ...\n";
            {
                int mid = N_SWEEP / 2;
                std::vector<EquilibriumResult> eq_results(N_SWEEP);
                eq_results[mid] = solve_equilibrium(p1_sweep[mid], p2_sweep[mid], false);
                for (int i = mid + 1; i < N_SWEEP; ++i)
                    eq_results[i] = solve_equilibrium_warm(
                        p1_sweep[i], p2_sweep[i],
                        eq_results[i - 1].D1, eq_results[i - 1].D2, false);
                for (int i = mid - 1; i >= 0; --i)
                    eq_results[i] = solve_equilibrium_warm(
                        p1_sweep[i], p2_sweep[i],
                        eq_results[i + 1].D1, eq_results[i + 1].D2, false);
                for (int i = 0; i < N_SWEEP; ++i) {
                    auto key = make_eq_key(p1_sweep[i], p2_sweep[i], 1, 2);
                    eq_cache.emplace(key, std::move(eq_results[i]));
                }
            }

            for (auto& cfg : alloc_configs) {
                std::cout << "  === " << rcfg.label << " targets=(" << cfg.b1 << "," << cfg.b2 << ") ===\n";
                g_b1 = cfg.b1;
                g_b2 = cfg.b2;

                // Full-info bar (mean-field) game: LQ tracking problem
                // Cost: (X̄ - b_i)² + r_i·D̄_i², dynamics: dX̄/dt = D̄₁ + D̄₂
                // Value function: V_i = P_i·X̄² + Q_i·X̄ + R_i
                // P_i: coupled Riccati (same as variance Riccati S_fi since running cost coeff = 1)
                // Q_i: backward linear ODE capturing target-tracking
                //   Q̇_i = 2b_i + P_i·Q_j/r_j + Q_i·(P_i/r_i + P_j/r_j),  Q_i(T)=0
                // Control: D̄_i = -(P_i/r_i)·X̄ - Q_i/(2r_i)
                // Note: P_i = S_fi (bar Riccati = variance Riccati for same running cost x²)

                // Step 1: Q_i backward (linear ODE driven by targets b_i)
                std::array<double, N_MAX> Q_fi1{}, Q_fi2{};
                for (int j = g_n - 2; j >= 0; --j) {
                    double qq1 = Q_fi1[j + 1], qq2 = Q_fi2[j + 1];
                    double p1 = S_fi1[j + 1], p2 = S_fi2[j + 1];
                    // Forward-time: Q̇_i = 2b_i + P_i·Q_j/r_j + Q_i·(P_i/r_i + P_j/r_j)
                    // Backward integration: Q[j] = Q[j+1] - dt·(...)
                    Q_fi1[j] = qq1 - g_dt * (2.0 * cfg.b1 + p1 * qq2 / rcfg.r2
                               + qq1 * (p1 / rcfg.r1 + p2 / rcfg.r2));
                    Q_fi2[j] = qq2 - g_dt * (2.0 * cfg.b2 + p2 * qq1 / rcfg.r1
                               + qq2 * (p2 / rcfg.r2 + p1 / rcfg.r1));
                }

                // Step 2: Forward simulate bar trajectory
                std::array<double, N_MAX> barX_fi{};
                std::array<double, N_MAX> barD1_fi{}, barD2_fi{};
                barX_fi[0] = X0;
                for (int j = 0; j < g_n; ++j) {
                    barD1_fi[j] = -(S_fi1[j] / rcfg.r1) * barX_fi[j] - Q_fi1[j] / (2.0 * rcfg.r1);
                    barD2_fi[j] = -(S_fi2[j] / rcfg.r2) * barX_fi[j] - Q_fi2[j] / (2.0 * rcfg.r2);
                    if (j < g_n - 1)
                        barX_fi[j + 1] = barX_fi[j] + g_dt * (barD1_fi[j] + barD2_fi[j]);
                }

                // Step 3: Compute full-info costs
                // J_i = variance part + bar part
                // Variance: ∫σ²S_i dt  (already computed as J_fi_var)
                // Bar: ∫[(X̄-b_i)² + r_i·D̄_i²] dt
                double j1_fi_bar = 0.0, j2_fi_bar = 0.0;
                double barD1sq_fi = 0.0, barD2sq_fi = 0.0;
                for (int j = 0; j < g_n; ++j) {
                    double dx1 = barX_fi[j] - cfg.b1;
                    double dx2 = barX_fi[j] - cfg.b2;
                    j1_fi_bar += g_dt * (dx1 * dx1 + rcfg.r1 * barD1_fi[j] * barD1_fi[j]);
                    j2_fi_bar += g_dt * (dx2 * dx2 + rcfg.r2 * barD2_fi[j] * barD2_fi[j]);
                    barD1sq_fi += barD1_fi[j] * barD1_fi[j] * g_dt;
                    barD2sq_fi += barD2_fi[j] * barD2_fi[j] * g_dt;
                }
                double j1_fi = J1_fi_var + j1_fi_bar;
                double j2_fi = J2_fi_var + j2_fi_bar;
                std::cout << "    Full-info: J=" << (j1_fi + j2_fi)
                          << " (var=" << (J1_fi_var + J2_fi_var) << " bar=" << (j1_fi_bar + j2_fi_bar)
                          << ") barD1sq=" << barD1sq_fi << " barD2sq=" << barD2sq_fi << "\n";

                for (int i = 0; i < N_SWEEP; ++i) {
                    double p1v = p1_sweep[i];
                    double p2v = p2_sweep[i];

                    // --- Equilibrium cost ---
                    auto& eq_a = cached_solve(p1v, p2v);
                    auto& bar_a = cached_bar_solve(eq_a, p1v*p1v, p2v*p2v,
                                                   p1v, p2v, 1, 2);
                    auto [j1_eq, j2_eq] = compute_costs_general(
                        eq_a.env, eq_a.calD1, eq_a.calD2, bar_a,
                        rcfg.r1, rcfg.r2, cfg.b1, cfg.b2);

                    // --- Compute wedges V1, V2 at terminal time ---
                    auto prec1_arr = make_constant_prec(p1v * p1v);
                    auto prec2_arr = make_constant_prec(p2v * p2v);
                    auto bba1 = backward_bar_adjoints(eq_a.env.X, eq_a.env.Xtilde2, eq_a.D2,
                                                      bar_a.barX, cfg.b1, prec2_arr, 0.0);
                    auto bba2 = backward_bar_adjoints(eq_a.env.X, eq_a.env.Xtilde1, eq_a.D1,
                                                      bar_a.barX, cfg.b2, prec1_arr, 0.0);

                    // Integrate wedges over time (sum across all t)
                    double V1_total = 0.0, V2_total = 0.0;
                    for (int j = 0; j < g_n; ++j) {
                        double v1_t = 0.0, v2_t = 0.0;
                        for (int z = 0; z <= j; ++z) {
                            v1_t += eq_a.env.Xtilde2[j][z].dot(bba1.barHk[j][z]);
                            v2_t += eq_a.env.Xtilde1[j][z].dot(bba2.barHk[j][z]);
                        }
                        V1_total += v1_t * (p2v * p2v) * g_dt;
                        V2_total += v2_t * (p1v * p1v) * g_dt;
                    }

                    // --- Mean control energy ---
                    double barD1sq_eq = 0.0, barD2sq_eq = 0.0;
                    for (int j = 0; j < g_n; ++j) {
                        barD1sq_eq += bar_a.barD1[j] * bar_a.barD1[j] * g_dt;
                        barD2sq_eq += bar_a.barD2[j] * bar_a.barD2[j] * g_dt;
                    }

                    f << cfg.label << "," << rcfg.label << "," << rcfg.r1 << "," << rcfg.r2 << ","
                      << p1v << "," << (p1v*p1v) << ","
                      << j1_eq << "," << j2_eq << "," << (j1_eq + j2_eq) << ","
                      << j1_fi << "," << j2_fi << "," << (j1_fi + j2_fi) << ","
                      << V1_total << "," << V2_total << ","
                      << barD1sq_eq << "," << barD2sq_eq << ","
                      << barD1sq_fi << "," << barD2sq_fi << "\n";

                    if (i % 15 == 0)
                        std::cout << "    p1=" << p1v << ": J_eq=" << (j1_eq + j2_eq)
                                  << " barD1sq=" << barD1sq_eq << " barD2sq=" << barD2sq_eq << "\n";
                }
            }
        }
    }

    // Restore defaults
    g_r1 = RHO;
    g_r2 = RHO;
    g_b1 = B1_DEFAULT;
    g_b2 = B2_DEFAULT;
    g_sigma = 1.0;

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "All data written to " << DATA_DIR << "/\n";
    std::cout << "Run: python3 plot_figures.py\n";

    return 0;
}
