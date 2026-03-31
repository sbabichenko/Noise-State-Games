#!/usr/bin/env python3
"""
Build interactive.html from interactive_template.html.

Note:
- solver.js may contain raw wasm bytes (NUL characters) when built with
  Emscripten single-file binary encoding.
- Inlining that payload into an HTML <script> block can corrupt the page in
  browsers, so we keep solver.js external and let the template load it at
  runtime.
"""

import os

DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR, "interactive_template.html"), "r") as f:
    template = f.read()

out_path = os.path.join(DIR, "interactive.html")
with open(out_path, "w") as f:
    f.write(template)

size_kb = os.path.getsize(out_path) / 1024
print(f"Built {out_path} ({size_kb:.0f} KB; loads solver.js externally)")
