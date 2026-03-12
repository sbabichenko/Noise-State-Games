# Decentralized LQG Noise-State Game Solver

Solver for Nash equilibria in two-player linear-quadratic-Gaussian games with
decentralized information. Each player privately observes one component of a
shared Brownian noise and must choose feedback controls without knowing the
other player's observation.

The solver computes infinite-dimensional kernel-valued fixed points
(feedback kernels **D**, state kernels **X**, filter kernels **R**) on
a discrete time grid, then derives mean-field trajectories and player costs.

## Quick start

```bash
# Dependencies: Eigen 3.3+, CMake 3.14+, a C++17 compiler, OpenMP (optional)
mkdir build && cd build
cmake .. && make -j$(nproc)
cd .. && ./build/generate_figures   # writes CSV to data/
python3 plot_figures.py             # renders Figures 3–12 as PDF
```

## Algorithm overview

The equilibrium is found by a three-level nested iteration:

1. **Picard / Anderson outer loop** (`solve_equilibrium`):
   update feedback kernels D₁, D₂ via the best-response map
   G(D) = −(1/ρ) H_x, accelerated with Anderson(5) mixing.

2. **Forward environment** (`forward_environment`):
   given D₁, D₂, iterate the coupled filter–control system
   for `FORWARD_INNER_ITERS` (= 5) steps to obtain the state kernel X,
   filter gains R, rank-1 factors A, and primitive control kernels calD.

3. **Rank-1 filter iteration** (inside `compute_filter_kernels`):
   solve for the filter kernel's A factor via
   `FILTER_INNER_ITERS` (= 8) relaxation steps per time index j.

After the kernel equilibrium converges, a separate scalar fixed-point
(`solve_bar_equilibrium`) finds the mean-field trajectory, and backward
adjoints compute cost sensitivities.

### F-free decomposition

The 3-D filter kernel F\[j\]\[u\]\[s\] (an N × N × N array of 3 × 3 matrices,
~36 MB) is never materialised during the solve.  Instead every product of
the form  Σ_u F\[j\]\[u\]\[s\]ᵀ v\[u\]  is evaluated directly from

```
F[j][u][s] = border(u, s) + Σ_{k>max(u,s)}  R[k][u] · A[k][s]ᵀ
```

using the already-available rank-1 factors R and A.  This eliminates
the dominant memory allocation and the O(N³) copy that was required to
populate F at every filter call.  F is only materialised once, for
Figure 7 output (`materialize_F`).

### Anderson acceleration

The outer Picard loop uses Anderson(m = 5) acceleration with damped
mixing (β = 0.6) and Tikhonov regularisation (λ = 10⁻¹²).
This typically reduces iteration count by 2–3× compared to simple
relaxation, converging in 19–47 iterations depending on problem
parameters.

## Time complexity

All complexities below use **N** = number of time grid points (default 40)
and **D** = state/noise dimension (fixed at 3).

| Function | Time per call | Calls per Picard iter |
|---|---|---|
| `compute_filter_kernels` | O(N³ D) | 2 × FORWARD_INNER_ITERS = 10 |
| `primitive_control_kernel` | O(N³ D) | 2 × FORWARD_INNER_ITERS = 10 |
| `state_kernel_from_calD` | O(N² D) | FORWARD_INNER_ITERS + 1 = 6 |
| `backward_kernels` | O(N³ D²) | 2 |
| Anderson Gram matrix | O(m² N² D) | 1 |

**Per Picard iteration**: O(N³ D) dominated by 20 filter/control kernel calls
inside `forward_environment`, plus 2 backward passes.

**Per equilibrium solve**: O(K × N³ D) where K ≈ 19–47 iterations.
With N = 40, D = 3, K ≈ 20: roughly 4 × 10⁹ scalar operations, completing
in ~0.02 s on a modern x86-64 core.

**Full figure generation** solves 15 unique equilibria (parallelised over
4 threads) plus ~25 bar-equilibrium solves and backward adjoints, totalling
~0.26 s wall-clock.

### How runtime scales with N

The solver is cubic in N.  Doubling N from 40 → 80 increases per-solve
time by roughly 8×.  The number of Picard iterations is largely
independent of N (determined by the spectral radius of the best-response
map), so total time scales as **O(N³)**.

## Memory usage

| Allocation | Size formula | N = 40 |
|---|---|---|
| Kernel2D | N² × D × 8 B | 38 KB |
| Kernel3D (F, only for figures) | N³ × D² × 8 B | 36 MB |
| HkSlice (backward ping-pong) | N² × D² × 8 B | 115 KB |
| EnvironmentResult | 10 × Kernel2D | 342 KB |
| Anderson history (ring of 5) | 5 × 4 × Kernel2D | 760 KB |

**Peak during equilibrium solve**: ~1.3 MB (no Kernel3D).

**Peak during figure generation**: ~37 MB (one Kernel3D for Figure 7,
allocated and freed in a local scope).

### How memory scales with N

Working memory (Kernel2D-based) scales as **O(N² D)**.
The one-time Kernel3D allocation for figure output scales as **O(N³ D²)**.
With D = 3 fixed, doubling N quadruples working memory and octuples the
figure-output allocation.

## Parallelisation

OpenMP is used at three levels:

1. **Outer**: independent equilibrium solves run in parallel via
   `#pragma omp parallel for` (up to 15 concurrent solves).
2. **Mid**: inside each Picard iteration, player-1 and player-2
   filter/control/backward computations run as `omp parallel sections`.
3. **Caching**: equilibrium and bar-solution results are cached with
   `omp critical` guards to avoid redundant solves across figures.

With 4 cores the 15 equilibria complete in ~0.17 s wall-clock
(vs ~0.58 s single-threaded).

## File layout

```
lqg_solver.h          type definitions, constants, function signatures
lqg_solver.cpp        solver implementation (all kernels, fixed-point loops)
generate_figures.cpp   driver: pre-solves equilibria, writes CSV to data/
plot_figures.py        matplotlib script: reads CSV, renders Figures 3–12
CMakeLists.txt         build configuration (C++17, Eigen3, OpenMP)
data/                  generated CSV files (created at runtime)
```

## Parameters

All solver constants are in `lqg_solver.h`:

| Constant | Default | Description |
|---|---|---|
| `N` | 40 | Time grid points |
| `T_VAL` | 1 | Planning horizon \[0, T\] |
| `RHO` | 0.1 | Control cost weight |
| `D_W` | 3 | Noise/state dimension |
| `FORWARD_INNER_ITERS` | 5 | Filter–control iterations per Picard step |
| `FILTER_INNER_ITERS` | 8 | Rank-1 filter sub-iterations |
| `PICARD_TOL` | 10⁻⁵ | Outer convergence tolerance |
| `MAX_PICARD_ITERS` | 10000 | Outer iteration cap |
