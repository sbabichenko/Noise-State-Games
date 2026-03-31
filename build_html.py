#!/usr/bin/env python3
"""
Build a self-contained interactive.html by inlining solver.js into the template.
Plotly.js is loaded from CDN (3.5MB is too large to inline).
"""

import os

DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR, "solver.js"), "r", encoding="utf-8") as f:
    solver_js = f.read()

with open(os.path.join(DIR, "interactive_template.html"), "r", encoding="utf-8") as f:
    template = f.read()

html = template.replace("/* SOLVER_JS_INLINE */", solver_js)

out_path = os.path.join(DIR, "interactive.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(out_path) / 1024
print(f"Built {out_path} ({size_kb:.0f} KB)")
