// ============================================================
// Generate all paper figures for the decentralized LQG game.
// Outputs CSV data files to data/ directory; plot with plot_figures.py
// ============================================================

#include "lqg_solver.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

static const std::string DATA_DIR = "data";

// ---------- CSV helpers ----------
// Write a 2D kernel (N x N x D_W) as CSV: columns t, s, ch0, ch1, ch2
// Only writes entries where s <= t (causal kernel)
static void write_kernel2d(const std::string& path, const Kernel2D& K) {
    std::ofstream f(path);
    f << std::setprecision(12);
    f << "t_idx,s_idx,t,s,ch0,ch1,ch2\n";
    const auto& tg = t_grid();
    for (int ti = 0; ti < N; ++ti)
        for (int si = 0; si <= ti; ++si)
            f << ti << "," << si << "," << tg[ti] << "," << tg[si] << ","
              << K[ti][si](0) << "," << K[ti][si](1) << "," << K[ti][si](2) << "\n";
}

// Write F1[T] kernel: F1[N-1][u][s](row, col) for all u, s, row, col
static void write_F1_T(const std::string& path, const Kernel3D& F1) {
    std::ofstream f(path);
    f << std::setprecision(12);
    f << "u_idx,s_idx,u,s,row,col,value\n";
    const auto& tg = t_grid();
    for (int u = 0; u < N; ++u)
        for (int s = 0; s < N; ++s)
            for (int row = 0; row < D_W; ++row)
                for (int col = 0; col < D_W; ++col)
                    f << u << "," << s << "," << tg[u] << "," << tg[s] << ","
                      << row << "," << col << "," << F1[N-1][u][s](row, col) << "\n";
}

// ---------- main ----------
int main() {
    fs::create_directories(DATA_DIR);
    const auto& tg = t_grid();

    // ============================================================
    // Solve symmetric benchmark p1 = p2 = 3
    // ============================================================
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Solving symmetric benchmark p1=p2=3 ...\n";

    auto eq = solve_equilibrium(3, 3);
    auto& D1 = eq.D1;
    auto& D2 = eq.D2;
    auto& env = eq.env;

    auto bar_sol = solve_bar_equilibrium(env, D1, D2, 9, 9, 500, 0.35, 1e-10);
    std::cout << "  Bar residual: " << bar_sol.bar_residual << "\n";

    // Backward bar adjoints for Figure 9
    auto prec9 = make_constant_prec(9.0);
    auto bba = backward_bar_adjoints(env.X, env.R2, D2, bar_sol.barX, g_b1, prec9, 0.0);

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
    write_kernel2d(DATA_DIR + "/fig6_calD1.csv", env.calD1);

    // ============================================================
    // FIGURE 7: F1 at t=T
    // ============================================================
    std::cout << "Figure 7: Filtering kernel F1 at t=T ...\n";
    {
        Kernel3D F1;
        materialize_F(env.R1, env.A_store1, env.obs_gain1, env.obs_idx1, F1);
        write_F1_T(DATA_DIR + "/fig7_F1_T.csv", F1);
    }

    // ============================================================
    // FIGURE 8: Mean control barD1(t) vs precision p
    // ============================================================
    std::cout << "Figure 8: Mean control vs precision ...\n";
    std::vector<int> p_values = {1, 2, 3, 5, 10};

    // Perfect-info benchmark
    std::array<double, N> S_pi;
    S_pi.fill(0.0);
    S_pi[N - 1] = TERMINAL_STATE_WEIGHT;
    for (int j = N - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1] + DT * (1.0 - (2.0 / RHO) * S_pi[j + 1] * S_pi[j + 1]);

    std::array<double, N> barD1_pi, barX_pi;
    barD1_pi.fill(0.0);
    barX_pi.fill(0.0);
    barX_pi[0] = X0;
    for (int j = 0; j < N; ++j) {
        barD1_pi[j] = -(1.0 / RHO) * S_pi[j] * (barX_pi[j] - B1_DEFAULT);
        double barD2_pi_j = -(1.0 / RHO) * S_pi[j] * (barX_pi[j] - B2_DEFAULT);
        if (j < N - 1)
            barX_pi[j + 1] = barX_pi[j] + DT * (barD1_pi[j] + barD2_pi_j);
    }

    {
        std::ofstream f(DATA_DIR + "/fig8_barD1.csv");
        f << std::setprecision(12);
        f << "t,perfect_info";
        for (int p : p_values)
            f << ",p" << p;
        f << "\n";

        std::map<int, std::array<double, N>> barD1_curves;

        for (int p : p_values) {
            std::cout << "  Solving p=" << p << " ...\n";
            auto eq_p = solve_equilibrium(p, p, false);
            auto bar_p = solve_bar_equilibrium(eq_p.env, eq_p.D1, eq_p.D2,
                                               p * p, p * p, 2000, 0.08, 1e-10);
            barD1_curves[p] = bar_p.barD1;
            std::cout << "    bar residual: " << bar_p.bar_residual << "\n";
        }

        for (int i = 0; i < N; ++i) {
            f << tg[i] << "," << barD1_pi[i];
            for (int p : p_values)
                f << "," << barD1_curves[p][i];
            f << "\n";
        }
    }

    // ============================================================
    // FIGURE 9: barH2 and R2
    // ============================================================
    std::cout << "Figure 9: Opponent adjoint barH2 and gain R2 ...\n";
    write_kernel2d(DATA_DIR + "/fig9_barH2.csv", bba.barHk);
    write_kernel2d(DATA_DIR + "/fig9_R2.csv", env.R2);

    // ============================================================
    // FIGURE 10: Asymmetric equilibrium panels (p1=3, p2 varies)
    // ============================================================
    std::cout << "Figure 10: Asymmetric equilibrium panels (p1=3, p2 varies) ...\n";
    int p1_fixed = 3;
    std::vector<int> p2_values = {1, 2, 3, 5, 10, 20};

    {
        std::ofstream f(DATA_DIR + "/fig10_asymmetric.csv");
        f << std::setprecision(12);
        f << "t,p2,barD1,barD2,barX\n";

        for (int p2v : p2_values) {
            std::cout << "  Solving p1=" << p1_fixed << ", p2=" << p2v << " ...\n";
            auto eq_a = solve_equilibrium(p1_fixed, p2v, false);
            auto bar_a = solve_bar_equilibrium(eq_a.env, eq_a.D1, eq_a.D2,
                                               p1_fixed * p1_fixed, p2v * p2v,
                                               2000, 0.08, 1e-10);
            std::cout << "    bar residual: " << bar_a.bar_residual << "\n";

            for (int i = 0; i < N; ++i)
                f << tg[i] << "," << p2v << ","
                  << bar_a.barD1[i] << "," << bar_a.barD2[i] << ","
                  << bar_a.barX[i] << "\n";
        }

        // Also write perfect-info effort benchmark
        std::ofstream fpi(DATA_DIR + "/fig10_perfect_info.csv");
        fpi << std::setprecision(12);
        fpi << "t,barD1_pi\n";
        for (int i = 0; i < N; ++i)
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
            auto eq_a = solve_equilibrium(p1_fixed, p2v, false);
            auto bar_a = solve_bar_equilibrium(eq_a.env, eq_a.D1, eq_a.D2,
                                               p1_fixed * p1_fixed, p2v * p2v,
                                               2000, 0.08, 1e-10);

            auto prec2_arr = make_constant_prec(static_cast<double>(p2v * p2v));
            auto prec1_arr = make_constant_prec(static_cast<double>(p1_fixed * p1_fixed));

            auto bba1 = backward_bar_adjoints(eq_a.env.X, eq_a.env.R2, eq_a.D2,
                                              bar_a.barX, B1_DEFAULT, prec2_arr, 0.0);
            auto bba2 = backward_bar_adjoints(eq_a.env.X, eq_a.env.R1, eq_a.D1,
                                              bar_a.barX, B2_DEFAULT, prec1_arr, 0.0);

            for (int j = 0; j < N; ++j) {
                int m = j + 1;
                double V1 = 0.0, V2 = 0.0;
                for (int z = 0; z < m; ++z) {
                    V1 += eq_a.env.R2[j][z].dot(bba1.barHk[j][z]);
                    V2 += eq_a.env.R1[j][z].dot(bba2.barHk[j][z]);
                }
                V1 *= static_cast<double>(p2v * p2v) * DT;
                V2 *= static_cast<double>(p1_fixed * p1_fixed) * DT;

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
                auto eq_a = solve_equilibrium(p1_fixed, p2v, false);
                auto bar_a = solve_bar_equilibrium(eq_a.env, eq_a.D1, eq_a.D2,
                                                   p1_fixed * p1_fixed, p2v * p2v,
                                                   2000, 0.08, 1e-10);
                auto [j1p, j2p_priv] = compute_costs_general(eq_a.env, bar_a, RHO, cfg.b1, cfg.b2);

                // Pooled
                double p_common = std::sqrt(p1_fixed * p1_fixed + p2v * p2v);
                std::cout << "    Pooled: p_common=" << p_common << " ...\n";
                auto eq_c = solve_equilibrium(p_common, p_common, false,
                                              Pi1(), 1, Pi1(), 1);
                auto bar_c = solve_bar_equilibrium(eq_c.env, eq_c.D1, eq_c.D2,
                                                   p_common * p_common, p_common * p_common,
                                                   2000, 0.08, 1e-10);
                auto [j1pool, j2pool] = compute_costs_general(eq_c.env, bar_c, RHO, cfg.b1, cfg.b2);

                f << cfg.label << "," << p2v << ","
                  << j1p << "," << j2p_priv << ","
                  << j1pool << "," << j2pool << "\n";
            }

            g_b1 = B1_DEFAULT;
            g_b2 = B2_DEFAULT;
        }
    }

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "All data written to " << DATA_DIR << "/\n";
    std::cout << "Run: python3 plot_figures.py\n";

    return 0;
}
