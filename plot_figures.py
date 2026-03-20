#!/usr/bin/env python3
"""
Plot all paper figures from CSV data produced by the C++ solver.
Reads from data/, writes PDFs to figs/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

DATA_DIR = 'data'
FIGDIR = 'figs'
os.makedirs(FIGDIR, exist_ok=True)

T_VAL = 1
N = 40
t_grid = np.linspace(0.0, T_VAL, N)
channel_labels = [r'$W^0$', r'$W^1$', r'$W^2$']


def load_kernel2d(path):
    """Load a 2D kernel CSV into (N, N, 3) array."""
    df = pd.read_csv(path)
    K = np.zeros((N, N, 3))
    for _, row in df.iterrows():
        ti, si = int(row['t_idx']), int(row['s_idx'])
        K[ti, si] = [row['ch0'], row['ch1'], row['ch2']]
    return K


def make_3channel_fig(data, title_prefix, clabel, n_curves=15):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.8))
    cmap_loc = cm.viridis
    norm_loc = Normalize(vmin=0, vmax=T_VAL)
    for ch in range(3):
        ax = axes[ch]
        max_val = np.max(np.abs(data[:, :, ch]))
        for t_idx in range(0, N, max(1, N // n_curves)):
            color = cmap_loc(norm_loc(t_grid[t_idx]))
            ax.plot(t_grid[:t_idx+1], data[t_idx, :t_idx+1, ch], color=color, lw=1.2)
        ax.set_xlabel(r'$s$')
        ax.set_title(f'{title_prefix} channel {channel_labels[ch]}'
                     f'\nmax$|.|$={max_val:.3g}')
        ax.grid(alpha=0.2)
    fig.subplots_adjust(right=0.88, top=0.82)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    sm = cm.ScalarMappable(cmap=cmap_loc, norm=norm_loc)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=clabel)
    return fig


# ============================================================
# FIGURE 3: Picard residual
# ============================================================
print("Figure 3: Picard residual ...")
df = pd.read_csv(f'{DATA_DIR}/fig3_residuals.csv')
fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
ax.semilogy(df['iteration'], df['residual'], lw=2, color='C0')
ax.set_xlabel('Iteration')
ax.set_ylabel('Relative residual')
ax.set_title('Outer Picard Residual')
ax.grid(alpha=0.3)
fig.savefig(f'{FIGDIR}/fig3_picard_residual.pdf')
plt.close(fig)

# ============================================================
# FIGURE 4: State kernel X(t,s)
# ============================================================
print("Figure 4: State kernel X(t,s) ...")
X = load_kernel2d(f'{DATA_DIR}/fig4_X.csv')
fig = make_3channel_fig(X, r'$X$', r'$t$')
fig.savefig(f'{FIGDIR}/fig4_state_kernel.pdf')
plt.close(fig)

# ============================================================
# FIGURE 5: D1(t,s) feedback kernel
# ============================================================
print("Figure 5: D1(t,s) feedback kernel ...")
D1 = load_kernel2d(f'{DATA_DIR}/fig5_D1.csv')
fig = make_3channel_fig(D1, r'$D^1$', r'$t$')
fig.savefig(f'{FIGDIR}/fig5_D1_kernel.pdf')
plt.close(fig)

# ============================================================
# FIGURE 6: calD1(t,s)
# ============================================================
print("Figure 6: calD1(t,s) primitive-noise kernel ...")
calD1 = load_kernel2d(f'{DATA_DIR}/fig6_calD1.csv')
fig = make_3channel_fig(calD1, r'$\mathcal{D}^1$', r'$t$')
fig.savefig(f'{FIGDIR}/fig6_calD1_kernel.pdf')
plt.close(fig)

# ============================================================
# FIGURE 7: F1 at t=T
# ============================================================
print("Figure 7: Filtering kernel F1 at t=T ...")
df = pd.read_csv(f'{DATA_DIR}/fig7_F1_T.csv')
F1_T = np.zeros((N, N, 3, 3))
for _, row in df.iterrows():
    u, s = int(row['u_idx']), int(row['s_idx'])
    r, c = int(row['row']), int(row['col'])
    F1_T[u, s, r, c] = row['value']

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
cmap_f = cm.viridis
norm_u = Normalize(vmin=0, vmax=T_VAL)
row_labels = [r'to $W^0$', r'to $W^1$', r'to $W^2$']
col_labels = [r'from $W^0$', r'from $W^1$', r'from $W^2$']

for row in range(3):
    for col in range(3):
        ax = axes[row][col]
        max_val = np.max(np.abs(F1_T[:, :, row, col]))
        for u_idx in range(0, N, max(1, N // 12)):
            color = cmap_f(norm_u(t_grid[u_idx]))
            ax.plot(t_grid[:N], F1_T[u_idx, :N, row, col], color=color, lw=1.0, alpha=0.8)
        if max_val < 1e-10:
            ax.text(0.5, 0.5, r'$\approx 0$', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
        if row == 0: ax.set_title(f'{col_labels[col]}', fontsize=10)
        if col == 0: ax.set_ylabel(row_labels[row], fontsize=10)
        ax.set_xlabel(r'$s$', fontsize=9)
        ax.grid(alpha=0.2)

fig.subplots_adjust(right=0.88, top=0.92)
cax = fig.add_axes([0.90, 0.08, 0.015, 0.8])
sm = cm.ScalarMappable(cmap=cmap_f, norm=norm_u)
sm.set_array([])
fig.colorbar(sm, cax=cax, label=r'$u$')
fig.suptitle(r'$F^1$: entry slices of $F_t(u,s)$ at $t=T$ (color $= u$)', fontsize=14, y=0.97)
fig.savefig(f'{FIGDIR}/fig7_F1_kernel.pdf')
plt.close(fig)

# ============================================================
# FIGURE 8: Mean control barD1(t) vs precision p
# ============================================================
print("Figure 8: Mean control vs precision ...")
df = pd.read_csv(f'{DATA_DIR}/fig8_barD1.csv')
p_values = [1, 2, 3, 5, 10]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
cmap_p = cm.plasma
norm_p = Normalize(vmin=min(p_values), vmax=max(p_values))

for p in p_values:
    color = cmap_p(norm_p(p))
    ax.plot(df['t'], df[f'p{p}'], lw=2, color=color, label=f'$p={p}$')

ax.plot(df['t'], df['perfect_info'], lw=2, ls='--', color='gray', alpha=0.7, label='Perfect info')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\bar{D}^1(t)$')
ax.set_title(r'Player 1 mean control path $\bar{D}^1(t)$ vs perfect info (color $= p$)')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig8_barD1_vs_p.pdf')
plt.close(fig)

# ============================================================
# FIGURE 9: barH2 and R2
# ============================================================
print("Figure 9: Opponent adjoint barH2 and gain R2 ...")
barH2 = load_kernel2d(f'{DATA_DIR}/fig9_barH2.csv')
R2 = load_kernel2d(f'{DATA_DIR}/fig9_R2.csv')
norm9 = Normalize(vmin=0, vmax=T_VAL)

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))

for ch in range(3):
    ax = axes[0][ch]
    max_val = np.max(np.abs(barH2[:, :, ch]))
    for t_idx in range(0, N, max(1, N // 12)):
        color = cm.viridis(norm9(t_grid[t_idx]))
        ax.plot(t_grid[:t_idx+1], barH2[t_idx, :t_idx+1, ch], color=color, lw=1.0)
    ax.set_xlabel(r'$s$')
    ax.set_title(r'$\bar{H}^2$ channel ' + channel_labels[ch] + f'\nmax$|.|$={max_val:.3g}')
    ax.grid(alpha=0.2)

for ch in range(3):
    ax = axes[1][ch]
    max_val = np.max(np.abs(R2[:, :, ch]))
    for t_idx in range(0, N, max(1, N // 12)):
        color = cm.viridis(norm9(t_grid[t_idx]))
        ax.plot(t_grid[:t_idx+1], R2[t_idx, :t_idx+1, ch], color=color, lw=1.0)
    ax.set_xlabel(r'$s$')
    ax.set_title(r'$R^2$ channel ' + channel_labels[ch] + f'\nmax$|.|$={max_val:.3g}')
    ax.grid(alpha=0.2)

fig.subplots_adjust(right=0.88, top=0.88, hspace=0.45)
cax = fig.add_axes([0.90, 0.08, 0.015, 0.8])
sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm9)
sm.set_array([])
fig.colorbar(sm, cax=cax, label=r'$t$')
fig.suptitle(r'Opponent objects: $\bar{H}^2$ (top) and $R^2$ (bottom)', fontsize=13, y=0.97)
fig.savefig(f'{FIGDIR}/fig9_barH2_R2.pdf')
plt.close(fig)

# ============================================================
# FIGURE 10: Asymmetric equilibrium panels
# ============================================================
print("Figure 10: Asymmetric equilibrium panels ...")
df = pd.read_csv(f'{DATA_DIR}/fig10_asymmetric.csv')
df_pi = pd.read_csv(f'{DATA_DIR}/fig10_perfect_info.csv')
p2_values = sorted(df['p2'].unique())

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
cmap_p2 = cm.viridis
norm_p2 = Normalize(vmin=min(p2_values), vmax=max(p2_values))

# Panel 1: mean controls
ax = axes[0]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'barD1'], color=c, lw=1.8, ls='-')
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'barD2'], color=c, lw=1.8, ls='--')
ax.plot([], [], color='gray', ls='-', lw=1.5, label=r'$\bar{D}^1$ (solid)')
ax.plot([], [], color='gray', ls='--', lw=1.5, label=r'$\bar{D}^2$ (dashed)')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$\bar{D}^i(t)$')
ax.set_title('Mean controls')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

# Panel 2: mean state
ax = axes[1]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'barX'], color=c, lw=1.8,
            label=f'$p_2={int(p2v)}$')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$\bar{X}(t)$')
ax.set_title('Mean state path')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Panel 3: aggregate effort
ax = axes[2]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    effort = np.abs(df.loc[mask, 'barD1'].values) + np.abs(df.loc[mask, 'barD2'].values)
    ax.plot(df.loc[mask, 't'], effort, color=c, lw=1.8)
effort_pi = 2 * np.abs(df_pi['barD1_pi'].values)
ax.plot(df_pi['t'], effort_pi, lw=1.5, ls='--', color='gray', alpha=0.7, label='Perfect info')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$|\bar{D}^1|+|\bar{D}^2|$')
ax.set_title('Aggregate mean effort')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.subplots_adjust(right=0.90)
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = cm.ScalarMappable(cmap=cmap_p2, norm=norm_p2)
sm.set_array([])
fig.colorbar(sm, cax=cax, label=r'$p_2$')
fig.suptitle(r'Asymmetric equilibrium: $p_1=3$ fixed, $p_2$ varies ($r=0.1$, $T=1$)',
             fontsize=14, y=1.01)
fig.savefig(f'{FIGDIR}/fig10_asymmetric_panels.pdf')
plt.close(fig)

# ============================================================
# FIGURE 11: Information wedges
# ============================================================
print("Figure 11: Information wedges ...")
df = pd.read_csv(f'{DATA_DIR}/fig11_wedges.csv')
p2_values = sorted(df['p2'].unique())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cmap_p2 = cm.viridis
norm_p2 = Normalize(vmin=min(p2_values), vmax=max(p2_values))

ax = axes[0]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'V1'], lw=2.2, color=c,
            label=f'$p_2={int(p2v)}$')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$t$', fontsize=13); ax.set_ylabel(r'$\mathcal{V}^1(t)$', fontsize=13)
ax.set_title(r'Player 1 wedge $\mathcal{V}^1(t)$', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'V2'], lw=2.2, color=c,
            label=f'$p_2={int(p2v)}$')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$t$', fontsize=13); ax.set_ylabel(r'$\mathcal{V}^2(t)$', fontsize=13)
ax.set_title(r'Player 2 wedge $\mathcal{V}^2(t)$', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle(r'Information wedges ($p_1=3$ fixed, $p_2$ varies)', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig11_info_wedges.pdf')
plt.close(fig)

# ============================================================
# FIGURE 12: Player costs — private vs pooled
# ============================================================
print("Figure 12: Player costs, private vs pooled ...")
df = pd.read_csv(f'{DATA_DIR}/fig12_costs.csv')

configs = [
    ('competitive', r'Competitive ($\theta=\pm 1$)'),
    ('cooperative', r'Cooperative ($\theta=0,0$)'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (cfg_key, label) in zip(axes, configs):
    sub = df[df['config'] == cfg_key].sort_values('p2')
    ax.plot(sub['p2'], sub['J1_priv'], 'o--', lw=2, ms=7, color='C0', label=r'$J^1$ private')
    ax.plot(sub['p2'], sub['J1_pool'], 's-', lw=2, ms=7, color='C0', alpha=0.7, label=r'$J^1$ pooled')
    ax.plot(sub['p2'], sub['J2_priv'], 'o--', lw=2, ms=7, color='C3', label=r'$J^2$ private')
    ax.plot(sub['p2'], sub['J2_pool'], 's-', lw=2, ms=7, color='C3', alpha=0.7, label=r'$J^2$ pooled')
    ax.set_xlabel(r'$p_2$', fontsize=13)
    ax.set_ylabel(r'Cost', fontsize=13)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle(r'Equilibrium costs: private vs pooled ($p_1=3$, $r=0.1$)', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig12_costs_private_vs_pooled.pdf')
plt.close(fig)

# ============================================================
# FIGURE 13: Precision allocation sweep
# ============================================================
print("Figure 13: Precision allocation ...")
df = pd.read_csv(f'{DATA_DIR}/fig13_precision_allocation.csv')

configs_alloc = [
    ('competitive', r'Competitive ($\theta_1\!=\!1,\;\theta_2\!=\!{-}1$)'),
    ('cooperative', r'Cooperative ($\theta_1\!=\!\theta_2\!=\!0$)'),
]

PBAR = 20.0
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (cfg_key, label) in zip(axes, configs_alloc):
    sub = df[df['config'] == cfg_key].sort_values('p1_root')
    p1 = sub['p1_root'].values
    p1_frac = sub['p1_prec'].values / PBAR  # fraction of total precision to player 1
    J_eq = sub['Jtotal_eq'].values
    J_ce = sub['Jtotal_ce'].values

    ax.plot(p1_frac, J_eq, lw=2.5, color='C0', label=r'Equilibrium $J^1\!+\!J^2$')
    ax.plot(p1_frac, J_ce, lw=2.0, ls='--', color='C1', alpha=0.8, label=r'CE (Riccati) $J^1\!+\!J^2$')

    # Mark optima
    idx_eq = np.argmin(J_eq)
    idx_ce = np.argmin(J_ce)
    ax.axvline(p1_frac[idx_eq], color='k', ls='--', lw=1.2, alpha=0.7,
               label=f'Eq. optimum $P^1/\\bar{{P}}\\!={p1_frac[idx_eq]:.2f}$')
    ax.axvline(p1_frac[idx_ce], color='gray', ls='--', lw=1.2, alpha=0.7,
               label=f'CE optimum $P^1/\\bar{{P}}\\!={p1_frac[idx_ce]:.2f}$')

    ax.set_xlabel(r'$P^1 / \bar{P}$', fontsize=13)
    ax.set_ylabel(r'$J^1 + J^2$', fontsize=13)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

fig.suptitle(r'Precision allocation: $P^1\!+\!P^2=\bar{P}=%g$, $r=0.1$, $\sigma=0.5$, $T=1$' % PBAR,
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig_precision_allocation.pdf')
plt.close(fig)

# ============================================================
# FIGURE 14: Mean control energy & wedge decomposition
# ============================================================
print("Figure 14: Mean control energy & wedges ...")
df14 = pd.read_csv(f'{DATA_DIR}/fig13_precision_allocation.csv')

# --- Panel layout: 2 rows x 3 cols ---
# Row 1: Asymmetric r (r1=0.05, r2=0.2) — the "starving" story
#   (a) Individual efforts barD1², barD2² for eq vs CE
#   (b) Total destructive effort eq vs CE
#   (c) Wedges V1, V2
# Row 2: Symmetric r (r1=0.1, r2=0.1)
#   (a) Individual efforts
#   (b) Total destructive effort
#   (c) Wedges V1, V2

r_cases = [
    ('r0.05_0.2', r'Asymmetric $r_1\!=\!0.05,\; r_2\!=\!0.2$'),
    ('r0.1_0.1',  r'Symmetric $r_1\!=\!r_2\!=\!0.1$'),
]

fig, axes = plt.subplots(2, 3, figsize=(17, 9))

for row, (rc_key, rc_label) in enumerate(r_cases):
    sub = df14[(df14['config'] == 'competitive') & (df14['r_config'] == rc_key)].sort_values('p1_root')
    p1_frac = sub['p1_prec'].values / PBAR

    # (a) Individual efforts
    ax = axes[row, 0]
    ax.plot(p1_frac, sub['barD1sq_eq'].values, lw=2.2, color='C0',
            label=r'$\int \bar{D}_1^2\, dt$ (eq)')
    ax.plot(p1_frac, sub['barD2sq_eq'].values, lw=2.2, color='C3',
            label=r'$\int \bar{D}_2^2\, dt$ (eq)')
    ax.plot(p1_frac, sub['barD1sq_ce'].values, lw=1.8, ls='--', color='C0', alpha=0.6,
            label=r'$\int \bar{D}_1^2\, dt$ (CE)')
    ax.plot(p1_frac, sub['barD2sq_ce'].values, lw=1.8, ls='--', color='C3', alpha=0.6,
            label=r'$\int \bar{D}_2^2\, dt$ (CE)')
    ax.set_xlabel(r'$P^1 / \bar{P}$', fontsize=12)
    ax.set_ylabel(r'Mean control energy', fontsize=12)
    ax.set_title(f'Individual efforts — {rc_label}', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

    # (b) Total destructive effort
    ax = axes[row, 1]
    total_eq = sub['barD1sq_eq'].values + sub['barD2sq_eq'].values
    total_ce = sub['barD1sq_ce'].values + sub['barD2sq_ce'].values
    ax.plot(p1_frac, total_eq, lw=2.5, color='C0', label='Equilibrium')
    ax.plot(p1_frac, total_ce, lw=2.0, ls='--', color='C1', alpha=0.8, label='CE (Riccati)')
    ax.fill_between(p1_frac, total_eq, total_ce, alpha=0.12, color='C1')
    ax.set_xlabel(r'$P^1 / \bar{P}$', fontsize=12)
    ax.set_ylabel(r'$\int (\bar{D}_1^2 + \bar{D}_2^2)\, dt$', fontsize=12)
    ax.set_title(f'Total destructive effort — {rc_label}', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Wedges
    ax = axes[row, 2]
    ax.plot(p1_frac, sub['V1_total'].values, lw=2.2, color='C0',
            label=r'$\mathcal{V}^1$ (player 1 wedge)')
    ax.plot(p1_frac, sub['V2_total'].values, lw=2.2, color='C3',
            label=r'$\mathcal{V}^2$ (player 2 wedge)')
    ax.plot(p1_frac, sub['V1_total'].values + sub['V2_total'].values,
            lw=1.8, ls=':', color='k', alpha=0.7, label=r'$\mathcal{V}^1 + \mathcal{V}^2$')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel(r'$P^1 / \bar{P}$', fontsize=12)
    ax.set_ylabel(r'Integrated wedge', fontsize=12)
    ax.set_title(f'Information wedges — {rc_label}', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

fig.suptitle(r'Competitive targets ($\theta_1\!=\!1,\;\theta_2\!=\!{-}1$): '
             r'$P^1\!+\!P^2\!=\!\bar{P}\!=\!%g$, $\sigma\!=\!0.5$' % PBAR,
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig_control_energy_wedges.pdf')
plt.close(fig)

# ============================================================
# FIGURE 15: Total effort eq/ce ratio across r configurations
# ============================================================
print("Figure 15: Effort ratio across r configs ...")

r_all = [
    ('r0.1_0.1',  r'$r\!=\!0.1$'),
    ('r0.05_0.2', r'$r_1\!=\!0.05, r_2\!=\!0.2$'),
    ('r0.5_0.5',  r'$r\!=\!0.5$'),
    ('r0.5_2.0',  r'$r_1\!=\!0.5, r_2\!=\!2$'),
    ('r1.0_1.0',  r'$r\!=\!1$'),
    ('r2.0_2.0',  r'$r\!=\!2$'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: total effort for each r config
ax = axes[0]
cmap_r = cm.coolwarm
for idx, (rc_key, rc_label) in enumerate(r_all):
    sub = df14[(df14['config'] == 'competitive') & (df14['r_config'] == rc_key)].sort_values('p1_root')
    p1_frac = sub['p1_prec'].values / PBAR
    total_eq = sub['barD1sq_eq'].values + sub['barD2sq_eq'].values
    color = cmap_r(idx / (len(r_all) - 1))
    ax.plot(p1_frac, total_eq, lw=2.2, color=color, label=rc_label)
ax.set_xlabel(r'$P^1 / \bar{P}$', fontsize=13)
ax.set_ylabel(r'$\int (\bar{D}_1^2 + \bar{D}_2^2)\, dt$', fontsize=12)
ax.set_title('Total destructive effort (equilibrium)', fontsize=12)
ax.legend(fontsize=8, loc='best')
ax.grid(alpha=0.3)

# Right panel: eq/ce ratio
ax = axes[1]
for idx, (rc_key, rc_label) in enumerate(r_all):
    sub = df14[(df14['config'] == 'competitive') & (df14['r_config'] == rc_key)].sort_values('p1_root')
    p1_frac = sub['p1_prec'].values / PBAR
    total_eq = sub['barD1sq_eq'].values + sub['barD2sq_eq'].values
    total_ce = sub['barD1sq_ce'].values + sub['barD2sq_ce'].values
    ratio = total_eq / total_ce
    color = cmap_r(idx / (len(r_all) - 1))
    ax.plot(p1_frac, ratio, lw=2.2, color=color, label=rc_label)
ax.axhline(1.0, color='k', lw=1, ls=':', alpha=0.5)
ax.set_xlabel(r'$P^1 / \bar{P}$', fontsize=13)
ax.set_ylabel('Eq / CE effort ratio', fontsize=12)
ax.set_title('Strategic moderation vs distortion', fontsize=12)
ax.legend(fontsize=8, loc='best')
ax.grid(alpha=0.3)

fig.suptitle(r'Competitive targets: destructive effort across control costs '
             r'($P^1\!+\!P^2\!=\!%g$, $\sigma\!=\!0.5$)' % PBAR,
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig_effort_ratio.pdf')
plt.close(fig)

print("\nAll figures saved to", FIGDIR)
