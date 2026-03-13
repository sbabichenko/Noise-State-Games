#!/usr/bin/env python3
"""
Interactive Plotly Dash app for the decentralized LQG noise-state game.
Sliders for p1, p2, b1, b2 re-run the C++ solver and update all figures.

Usage:
    python3 interactive_app.py
    # Then open http://127.0.0.1:8050
"""

import json
import os
import subprocess
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, callback, dcc, html

N = 40
SOLVER = os.path.join(os.path.dirname(__file__), "build", "solve_interactive")

# Sweep grid for Figures 8 and 12
SWEEP_P2 = [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20]

# --------------- solver interface ---------------

def run_single(p1, p2, b1, b2, r1=0.1, r2=0.1):
    result = subprocess.run(
        [SOLVER, "single", str(p1), str(p2), str(b1), str(b2), str(r1), str(r2)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Solver failed: {result.stderr}")
    return json.loads(result.stdout)


def run_sweep(p1, b1, b2, r1=0.1, r2=0.1, p2_values=SWEEP_P2):
    cmd = [SOLVER, "sweep", str(p1), str(b1), str(b2), str(r1), str(r2)]
    cmd += [str(p) for p in p2_values]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Sweep failed: {result.stderr}")
    return json.loads(result.stdout)


def kernel_to_array(kernel_dict):
    arr = np.zeros((N, N, 3))
    for ch in range(3):
        arr[:, :, ch] = np.array(kernel_dict[f"ch{ch}"])
    return arr


# --------------- color helpers ---------------

def viridis_color(frac):
    import colorsys
    h = 0.75 - 0.55 * frac
    s = 0.7 + 0.2 * (1 - abs(2 * frac - 1))
    v = 0.5 + 0.4 * frac
    r, g, b = colorsys.hsv_to_rgb(max(0, min(1, h)), s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


VIRIDIS_COLORS = [viridis_color(i / (N - 1)) for i in range(N)]

# Plasma-ish scale for sweep curves (p2 coloring)
def plasma_color(frac):
    import colorsys
    h = 0.83 - 0.75 * frac  # purple → yellow
    s = 0.85
    v = 0.5 + 0.45 * frac
    r, g, b = colorsys.hsv_to_rgb(max(0, min(1, h)), s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


# --------------- single-solve figure builders ---------------

def make_kernel_fig(t, kernel_arr, title, n_curves=15):
    ch_labels = ["W⁰", "W¹", "W²"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        f"{title} channel {ch_labels[ch]}" for ch in range(3)
    ])
    step = max(1, N // n_curves)
    for ch in range(3):
        for ti in range(0, N, step):
            fig.add_trace(go.Scatter(
                x=t[:ti + 1], y=kernel_arr[ti, :ti + 1, ch],
                mode="lines", line=dict(color=VIRIDIS_COLORS[ti], width=1.5),
                showlegend=False,
                hovertemplate=f"t={t[ti]:.3f}<br>s=%{{x:.3f}}<br>val=%{{y:.4g}}",
            ), row=1, col=ch + 1)
    fig.update_xaxes(title_text="s")
    fig.update_layout(height=350, margin=dict(t=40, b=40, l=50, r=20))
    return fig


def make_residual_fig(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(data["residuals"]))), y=data["residuals"],
        mode="lines", line=dict(width=2),
    ))
    fig.update_yaxes(type="log", title_text="Relative residual")
    fig.update_xaxes(title_text="Iteration")
    fig.update_layout(
        title=f"Picard Residual ({data['n_iters']} iterations)",
        height=300, margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def make_controls_fig(data):
    t = data["t"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "Mean Controls", "Mean State Path", "Aggregate Effort"
    ])
    fig.add_trace(go.Scatter(x=t, y=data["barD1"], name="D̄¹",
                             line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=data["barD2"], name="D̄²",
                             line=dict(width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=data["barD1_pi"], name="Perfect info",
                             line=dict(width=1.5, dash="dot", color="gray")), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=data["barX"], name="X̄",
                             line=dict(width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=data["barX_pi"], name="X̄ perfect",
                             line=dict(width=1.5, dash="dot", color="gray"),
                             showlegend=False), row=1, col=2)
    fig.add_hline(y=data["b1"], line_dash="dot", line_color="rgba(0,0,0,0.3)",
                  annotation_text=f"b₁={data['b1']:.1f}", row=1, col=2)
    fig.add_hline(y=data["b2"], line_dash="dot", line_color="rgba(0,0,0,0.3)",
                  annotation_text=f"b₂={data['b2']:.1f}", row=1, col=2)

    effort = np.abs(data["barD1"]) + np.abs(data["barD2"])
    effort_pi = 2 * np.abs(np.array(data["barD1_pi"]))
    fig.add_trace(go.Scatter(x=t, y=effort.tolist(), name="|D̄¹|+|D̄²|",
                             line=dict(width=2), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=t, y=effort_pi.tolist(), name="Perfect info",
                             line=dict(width=1.5, dash="dot", color="gray"),
                             showlegend=False), row=1, col=3)
    fig.update_xaxes(title_text="t")
    fig.update_layout(height=350, margin=dict(t=40, b=40, l=50, r=20))
    return fig


def make_wedges_fig(data):
    t = data["t"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Player 1 wedge V¹(t)", "Player 2 wedge V²(t)"
    ])
    fig.add_trace(go.Scatter(x=t, y=data["V1"], name="V¹",
                             line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=data["V2"], name="V²",
                             line=dict(width=2)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_xaxes(title_text="t")
    fig.update_layout(height=300, margin=dict(t=40, b=40, l=50, r=20))
    return fig


# --------------- sweep figure builders ---------------

def make_barD1_sweep_fig(sweep_data):
    """Fig 8: barD1(t) curves for each p2 in sweep, plus perfect info."""
    t = sweep_data["t"]
    sweeps = sweep_data["sweeps"]
    p2_vals = [s["p2"] for s in sweeps]
    p2_min, p2_max = min(p2_vals), max(p2_vals)

    fig = go.Figure()
    for s in sweeps:
        frac = (s["p2"] - p2_min) / max(1, p2_max - p2_min)
        fig.add_trace(go.Scatter(
            x=t, y=s["barD1"], name=f"p₂={s['p2']:.1f}",
            line=dict(width=2, color=plasma_color(frac)),
            hovertemplate=f"p₂={s['p2']:.1f}<br>t=%{{x:.3f}}<br>D̄¹=%{{y:.4g}}",
        ))
    fig.add_trace(go.Scatter(
        x=t, y=sweep_data["barD1_pi"], name="Perfect info",
        line=dict(width=2, dash="dash", color="gray"),
    ))
    fig.update_xaxes(title_text="t")
    fig.update_yaxes(title_text="D̄¹(t)")
    fig.update_layout(
        title=f"Fig 8: Mean Control D̄¹(t) — p₁={sweep_data['p1']:.1f} fixed, p₂ varies",
        height=400, margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(x=1.02, y=1, font=dict(size=10)),
    )
    return fig


def make_barD2_sweep_fig(sweep_data):
    """barD2(t) curves for each p2 in sweep, plus perfect info."""
    t = sweep_data["t"]
    sweeps = sweep_data["sweeps"]
    p2_vals = [s["p2"] for s in sweeps]
    p2_min, p2_max = min(p2_vals), max(p2_vals)

    fig = go.Figure()
    for s in sweeps:
        frac = (s["p2"] - p2_min) / max(1, p2_max - p2_min)
        fig.add_trace(go.Scatter(
            x=t, y=s["barD2"], name=f"p\u2082={s['p2']:.1f}",
            line=dict(width=2, color=plasma_color(frac)),
            hovertemplate=f"p\u2082={s['p2']:.1f}<br>t=%{{x:.3f}}<br>D\u0304\u00b2=%{{y:.4g}}",
        ))
    fig.add_trace(go.Scatter(
        x=t, y=sweep_data["barD2_pi"], name="Perfect info",
        line=dict(width=2, dash="dash", color="gray"),
    ))
    fig.update_xaxes(title_text="t")
    fig.update_yaxes(title_text="D\u0304\u00b2(t)")
    fig.update_layout(
        title=f"Mean Control D\u0304\u00b2(t) \u2014 p\u2081={sweep_data['p1']:.1f} fixed, p\u2082 varies",
        height=400, margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(x=1.02, y=1, font=dict(size=10)),
    )
    return fig


def make_costs_fig(sweep_data):
    """Fig 12: Private vs pooled costs as p2 varies."""
    sweeps = sweep_data["sweeps"]
    p2_vals = [s["p2"] for s in sweeps]
    J1_priv = [s["J1_priv"] for s in sweeps]
    J2_priv = [s["J2_priv"] for s in sweeps]
    J1_pool = [s["J1_pool"] for s in sweeps]
    J2_pool = [s["J2_pool"] for s in sweeps]

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Player 1 Cost J¹", "Player 2 Cost J²"
    ])

    # Player 1
    fig.add_trace(go.Scatter(
        x=p2_vals, y=J1_priv, name="J¹ private", mode="lines+markers",
        line=dict(width=2, color="#1f77b4"), marker=dict(size=7),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=p2_vals, y=J1_pool, name="J¹ pooled", mode="lines+markers",
        line=dict(width=2, color="#1f77b4", dash="dash"), marker=dict(size=7, symbol="square"),
    ), row=1, col=1)

    # Player 2
    fig.add_trace(go.Scatter(
        x=p2_vals, y=J2_priv, name="J² private", mode="lines+markers",
        line=dict(width=2, color="#d62728"), marker=dict(size=7),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=p2_vals, y=J2_pool, name="J² pooled", mode="lines+markers",
        line=dict(width=2, color="#d62728", dash="dash"), marker=dict(size=7, symbol="square"),
    ), row=1, col=2)

    fig.update_xaxes(title_text="p₂")
    fig.update_yaxes(title_text="Cost")
    fig.update_layout(
        title=(f"Fig 12: Equilibrium Costs — p₁={sweep_data['p1']:.1f}, "
               f"b₁={sweep_data['b1']:.1f}, b₂={sweep_data['b2']:.1f} "
               f"(solid=private, dashed=pooled)"),
        height=400, margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def make_wedges_sweep_fig(sweep_data):
    """Fig 11: Information wedges V1(t), V2(t) for each p2 in sweep."""
    t = sweep_data["t"]
    sweeps = sweep_data["sweeps"]
    p2_vals = [s["p2"] for s in sweeps]
    p2_min, p2_max = min(p2_vals), max(p2_vals)

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Player 1 wedge V¹(t)", "Player 2 wedge V²(t)"
    ])
    for s in sweeps:
        frac = (s["p2"] - p2_min) / max(1, p2_max - p2_min)
        color = plasma_color(frac)
        fig.add_trace(go.Scatter(
            x=t, y=s["V1"], name=f"p₂={s['p2']:.1f}",
            line=dict(width=1.8, color=color), showlegend=True,
            legendgroup=f"p2_{s['p2']}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=s["V2"], name=f"p₂={s['p2']:.1f}",
            line=dict(width=1.8, color=color), showlegend=False,
            legendgroup=f"p2_{s['p2']}",
        ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_xaxes(title_text="t")
    fig.update_layout(
        title=f"Fig 11: Information Wedges — p₁={sweep_data['p1']:.1f} fixed",
        height=380, margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(x=1.02, y=1, font=dict(size=10)),
    )
    return fig


# --------------- app layout ---------------

app = Dash(__name__)
app.title = "LQG Noise-State Game Explorer"

slider_style = {"width": "100%", "padding": "0 10px"}

app.layout = html.Div([
    html.H2("Decentralized LQG Noise-State Game Explorer",
            style={"textAlign": "center", "marginBottom": "5px"}),

    # Parameter sliders
    html.Div([
        html.Div([
            html.Label("p₁ (player 1 precision)"),
            dcc.Slider(id="p1", min=0.5, max=20, step=0.5, value=3,
                       marks={i: str(i) for i in [1, 3, 5, 10, 15, 20]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"flex": "1", **slider_style}),
        html.Div([
            html.Label("p₂ (player 2 precision)"),
            dcc.Slider(id="p2", min=0.5, max=20, step=0.5, value=3,
                       marks={i: str(i) for i in [1, 3, 5, 10, 15, 20]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"flex": "1", **slider_style}),
        html.Div([
            html.Label("b₁ (player 1 target)"),
            dcc.Slider(id="b1", min=-3, max=3, step=0.1, value=1,
                       marks={i: str(i) for i in [-3, -2, -1, 0, 1, 2, 3]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"flex": "1", **slider_style}),
        html.Div([
            html.Label("b₂ (player 2 target)"),
            dcc.Slider(id="b2", min=-3, max=3, step=0.1, value=-1,
                       marks={i: str(i) for i in [-3, -2, -1, 0, 1, 2, 3]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"flex": "1", **slider_style}),
        html.Div([
            html.Label("r₁ (player 1 control cost)"),
            dcc.Slider(id="r1", min=0.01, max=1, step=0.01, value=0.1,
                       marks={v: str(v) for v in [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"flex": "1", **slider_style}),
        html.Div([
            html.Label("r₂ (player 2 control cost)"),
            dcc.Slider(id="r2", min=0.01, max=1, step=0.01, value=0.1,
                       marks={v: str(v) for v in [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"flex": "1", **slider_style}),
    ], style={"display": "flex", "gap": "10px", "margin": "10px 20px", "flexWrap": "wrap"}),

    # Status bar
    html.Div(id="status", style={
        "textAlign": "center", "padding": "5px",
        "fontFamily": "monospace", "fontSize": "13px", "color": "#555",
    }),

    # --- Sweep figures (depend on p1, b1, b2 only) ---
    html.H3("Sweep over p₂", style={"textAlign": "center", "marginTop": "20px",
                                      "color": "#333", "borderTop": "2px solid #ddd",
                                      "paddingTop": "15px"}),
    html.Div(id="sweep-status", style={
        "textAlign": "center", "padding": "3px",
        "fontFamily": "monospace", "fontSize": "12px", "color": "#888",
    }),
    dcc.Graph(id="fig-costs"),
    dcc.Graph(id="fig-barD1-sweep"),
    dcc.Graph(id="fig-barD2-sweep"),
    dcc.Graph(id="fig-wedges-sweep"),

    # --- Single-solve figures (depend on all 4 params) ---
    html.H3("Single Equilibrium Detail", style={
        "textAlign": "center", "marginTop": "20px", "color": "#333",
        "borderTop": "2px solid #ddd", "paddingTop": "15px",
    }),
    dcc.Graph(id="fig-residual"),
    dcc.Graph(id="fig-controls"),
    dcc.Graph(id="fig-X"),
    dcc.Graph(id="fig-D1"),
    dcc.Graph(id="fig-D2"),
    dcc.Graph(id="fig-calD1"),
    dcc.Graph(id="fig-wedges"),

], style={"maxWidth": "1400px", "margin": "0 auto", "fontFamily": "sans-serif"})


# --------------- callbacks ---------------

@callback(
    Output("fig-costs", "figure"),
    Output("fig-barD1-sweep", "figure"),
    Output("fig-barD2-sweep", "figure"),
    Output("fig-wedges-sweep", "figure"),
    Output("sweep-status", "children"),
    Input("p1", "value"),
    Input("b1", "value"),
    Input("b2", "value"),
    Input("r1", "value"),
    Input("r2", "value"),
)
def update_sweep_figures(p1, b1, b2, r1, r2):
    t0 = time.perf_counter()
    try:
        sweep = run_sweep(p1, b1, b2, r1, r2)
    except Exception as e:
        empty = go.Figure()
        return empty, empty, empty, empty, f"Sweep error: {e}"

    elapsed = time.perf_counter() - t0
    fig_costs = make_costs_fig(sweep)
    fig_barD1 = make_barD1_sweep_fig(sweep)
    fig_barD2 = make_barD2_sweep_fig(sweep)
    fig_wedges = make_wedges_sweep_fig(sweep)

    n_pts = len(sweep["sweeps"])
    status = f"Sweep: {n_pts} p\u2082 values solved in {elapsed:.3f}s"
    return fig_costs, fig_barD1, fig_barD2, fig_wedges, status


@callback(
    Output("fig-residual", "figure"),
    Output("fig-controls", "figure"),
    Output("fig-X", "figure"),
    Output("fig-D1", "figure"),
    Output("fig-D2", "figure"),
    Output("fig-calD1", "figure"),
    Output("fig-wedges", "figure"),
    Output("status", "children"),
    Input("p1", "value"),
    Input("p2", "value"),
    Input("b1", "value"),
    Input("b2", "value"),
    Input("r1", "value"),
    Input("r2", "value"),
)
def update_single_figures(p1, p2, b1, b2, r1, r2):
    t0 = time.perf_counter()
    try:
        data = run_single(p1, p2, b1, b2, r1, r2)
    except Exception as e:
        empty = go.Figure()
        return empty, empty, empty, empty, empty, empty, empty, f"Error: {e}"

    t = np.array(data["t"])
    elapsed = time.perf_counter() - t0

    fig_resid = make_residual_fig(data)
    fig_controls = make_controls_fig(data)
    fig_X = make_kernel_fig(t, kernel_to_array(data["X"]), "X(t,s)")
    fig_D1 = make_kernel_fig(t, kernel_to_array(data["D1"]), "D¹(t,s)")
    fig_D2 = make_kernel_fig(t, kernel_to_array(data["D2"]), "D²(t,s)")
    fig_calD1 = make_kernel_fig(t, kernel_to_array(data["calD1"]), "𝒟¹(t,s)")
    fig_wedges = make_wedges_fig(data)

    status = (
        f"p₁={p1} p₂={p2} b₁={b1} b₂={b2} r₁={r1} r₂={r2} | "
        f"{data['n_iters']} iters | "
        f"J¹={data['J1']:.4f} J²={data['J2']:.4f} | "
        f"bar resid={data['bar_residual']:.2e} | "
        f"solved in {elapsed:.3f}s"
    )
    return fig_resid, fig_controls, fig_X, fig_D1, fig_D2, fig_calD1, fig_wedges, status


if __name__ == "__main__":
    print("Starting LQG Game Explorer at http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
