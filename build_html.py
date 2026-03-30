#!/usr/bin/env python3
"""
Build a self-contained interactive.html by inlining solver.js into the template.
Plotly.js is loaded from CDN (3.5MB is too large to inline).

Important: solver.js may contain raw binary bytes (embedded WASM), including
NUL characters. Directly pasting those bytes into an HTML <script> block can
break parsing in some browsers. We therefore escape bytes into a JS string and
eval it at runtime.
"""

import os

DIR = os.path.dirname(os.path.abspath(__file__))

def bytes_to_js_string_literal(data: bytes) -> str:
    """Return a double-quoted JS string literal preserving raw byte values."""
    out = ['"']
    for b in data:
        # Safe printable ASCII except backslash and quote.
        if 0x20 <= b <= 0x7E and b not in (0x22, 0x5C):
            out.append(chr(b))
        elif b == 0x0A:
            out.append("\\n")
        elif b == 0x0D:
            out.append("\\r")
        elif b == 0x09:
            out.append("\\t")
        else:
            out.append(f"\\x{b:02x}")
    out.append('"')
    return "".join(out)


with open(os.path.join(DIR, "solver.js"), "rb") as f:
    solver_js_bytes = f.read()

solver_inline = (
    "(() => {\n"
    "  const __solver_src = " + bytes_to_js_string_literal(solver_js_bytes) + ";\n"
    "  (0, eval)(__solver_src);\n"
    "})();"
)

with open(os.path.join(DIR, "interactive_template.html"), "r") as f:
    template = f.read()

html = template.replace("/* SOLVER_JS_INLINE */", solver_inline)

out_path = os.path.join(DIR, "interactive.html")
with open(out_path, "w") as f:
    f.write(html)

size_kb = os.path.getsize(out_path) / 1024
print(f"Built {out_path} ({size_kb:.0f} KB)")
