#!/usr/bin/env python3
"""
Build a self-contained interactive.html by embedding solver.js safely.

solver.js may contain raw bytes (including NUL and '</script>' sequences), so
we never splice the raw text directly into HTML. Instead we embed base64 and
bootstrap it through a Blob URL.
"""

import os
import base64

DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR, "solver.js"), "rb") as f:
    solver_js_b64 = base64.b64encode(f.read()).decode("ascii")

with open(os.path.join(DIR, "interactive_template.html"), "r") as f:
    template = f.read()

inline_loader = f"""
(function() {{
  const b64 = "{solver_js_b64}";
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], {{ type: 'text/javascript' }});
  const url = URL.createObjectURL(blob);
  const script = document.createElement('script');
  script.src = url;
  script.onload = () => URL.revokeObjectURL(url);
  script.onerror = () => URL.revokeObjectURL(url);
  document.head.appendChild(script);
}})();
""".strip()

html = template.replace("/* SOLVER_JS_INLINE */", inline_loader)

out_path = os.path.join(DIR, "interactive.html")
with open(out_path, "w") as f:
    f.write(html)

size_kb = os.path.getsize(out_path) / 1024
print(f"Built {out_path} ({size_kb:.0f} KB; solver.js embedded as base64 blob)")
