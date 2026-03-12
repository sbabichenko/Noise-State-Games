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

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, callback, dcc, html

N = 40
SOLVER = os.path.join(os.path.dirname(__file__), "build", "solve_interactive")

# --------------- solver interface ---------------

def run_solver(p1, p2, b1, b2):
    result = subprocess.run(
        [SOLVER, str(p1), str(p2), str(b1), str(b2)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Solver failed: {result.stderr}")
    return json.loads(result.stdout)


def kernel_to_array(kernel_dict):
    """Convert {ch0: [[...]], ch1: ..., ch2: ...} to (N, N, 3) numpy array."""
    arr = np.zeros((N, N, 3))
    for ch in range(3):
        arr[:, :, ch] = np.array(kernel_dict[f"ch{ch}"])
    return arr


# --------------- color scales ---------------

def viridis_color(frac):
    """Map fraction [0,1] to a viridis-ish hex color."""
    import colorsys
    # Simple approximation: blue → teal → yellow
    h = 0.75 - 0.55 * frac  # hue from blue to yellow-green
    s = 0.7 + 0.2 * (1 - abs(2 * frac - 1))
    v = 0.5 + 0.4 * frac
    r, g, b = colorsys.hsv_to_rgb(max(0, min(1, h)), s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


VIRIDIS_COLORS = [viridis_color(i / (N - 1)) for i in range(N)]


# --------------- figure builders ---------------

def make_kernel_fig(t, kernel_arr, title, n_curves=15):
    """3-channel kernel plot with colored t-slices."""
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
                showlegend=False, hovertemplate=f"t={t[ti]:.3f}<br>s=%{{x:.3f}}<br>val=%{{y:.4g}}",
            ), row=1, col=ch + 1)
    fig.update_xaxes(title_text="s")
    fig.update_layout(height=350, margin=dict(t=40, b=40, l=50, r=20))
    return fig


def make_residual_fig(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(data["residuals"]))),
        y=data["residuals"],
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

    # Panel 1: barD1, barD2
    fig.add_trace(go.Scatter(x=t, y=data["barD1"], name="D̄¹",
                             line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=data["barD2"], name="D̄²",
                             line=dict(width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=data["barD1_pi"], name="Perfect info",
                             line=dict(width=1.5, dash="dot", color="gray")), row=1, col=1)

    # Panel 2: barX
    fig.add_trace(go.Scatter(x=t, y=data["barX"], name="X̄",
                             line=dict(width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=data["barX_pi"], name="X̄ perfect",
                             line=dict(width=1.5, dash="dot", color="gray"),
                             showlegend=False), row=1, col=2)
    fig.add_hline(y=data["b1"], line_dash="dot", line_color="rgba(0,0,0,0.3)",
                  annotation_text=f"b₁={data['b1']:.1f}", row=1, col=2)
    fig.add_hline(y=data["b2"], line_dash="dot", line_color="rgba(0,0,0,0.3)",
                  annotation_text=f"b₂={data['b2']:.1f}", row=1, col=2)

    # Panel 3: aggregate effort
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
    ], style={"display": "flex", "gap": "10px", "margin": "10px 20px"}),

    # Status bar
    html.Div(id="status", style={
        "textAlign": "center", "padding": "5px",
        "fontFamily": "monospace", "fontSize": "13px", "color": "#555",
    }),

    # Figures
    dcc.Graph(id="fig-residual"),
    dcc.Graph(id="fig-controls"),
    dcc.Graph(id="fig-X"),
    dcc.Graph(id="fig-D1"),
    dcc.Graph(id="fig-D2"),
    dcc.Graph(id="fig-calD1"),
    dcc.Graph(id="fig-wedges"),

], style={"maxWidth": "1400px", "margin": "0 auto", "fontFamily": "sans-serif"})


# --------------- callback ---------------

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
)
def update_figures(p1, p2, b1, b2):
    import time
    t0 = time.perf_counter()

    try:
        data = run_solver(p1, p2, b1, b2)
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
        f"p₁={p1} p₂={p2} b₁={b1} b₂={b2} | "
        f"{data['n_iters']} iters | "
        f"J¹={data['J1']:.4f} J²={data['J2']:.4f} | "
        f"bar resid={data['bar_residual']:.2e} | "
        f"solved in {elapsed:.3f}s"
    )

    return fig_resid, fig_controls, fig_X, fig_D1, fig_D2, fig_calD1, fig_wedges, status


if __name__ == "__main__":
    print("Starting LQG Game Explorer at http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
