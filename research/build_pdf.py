#!/usr/bin/env python3
"""Render the research paper markdown -> self-contained HTML (with inline
Mermaid) -> PDF via headless Chrome."""
import re, sys, pathlib, html as htmllib
from markdown_it import MarkdownIt

ROOT = pathlib.Path("/Users/alexmkwizu/Documents/SoftwareProjects/brush/research")
MD = ROOT / "brush-optimization-paper.md"
HTML = ROOT / "brush-optimization-paper.html"
MERMAID_JS = pathlib.Path("/tmp/mermaid.min.js").read_text()

text = MD.read_text()

# Pull out ```mermaid fences first so markdown-it doesn't escape them.
mermaid_blocks = []
def stash(m):
    mermaid_blocks.append(m.group(1))
    return f"\n@@MERMAID{len(mermaid_blocks)-1}@@\n"
text = re.sub(r"```mermaid\n(.*?)```", stash, text, flags=re.S)

# Trivial inline math: $...$ -> italic; tidy a superscript.
text = text.replace("(d+1)^2", "(d+1)²")
text = re.sub(r"\$([^$\n]+)\$", lambda m: f"*{m.group(1)}*", text)

md = MarkdownIt("commonmark", {"html": True}).enable("table")
body = md.render(text)

# Swap mermaid placeholders back in as <pre class="mermaid"> (Mermaid reads it).
def unstash(m):
    code = htmllib.escape(mermaid_blocks[int(m.group(1))])
    return f'<pre class="mermaid">{code}</pre>'
body = re.sub(r"<p>@@MERMAID(\d+)@@</p>", unstash, body)
body = re.sub(r"@@MERMAID(\d+)@@", unstash, body)

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: Georgia,'Times New Roman',serif; font-size: 10.5pt; line-height: 1.5;
  color:#111; max-width: 720px; margin: 0 auto; text-align: justify; }
h1 { font-size: 19pt; line-height:1.25; text-align:center; margin: 0 0 .2em; }
h2 { font-size: 13pt; border-bottom:1px solid #ccc; padding-bottom:2px; margin-top:1.4em; }
h3 { font-size: 11.5pt; margin-top:1.1em; }
h1 + p { text-align:center; }
p { margin: .5em 0; }
em { font-style: italic; }
code { font-family: 'SF Mono',Menlo,Consolas,monospace; font-size: 9pt;
  background:#f4f4f4; padding:1px 3px; border-radius:3px; }
pre code { background:none; padding:0; }
table { border-collapse: collapse; width:100%; margin: .8em 0; font-size: 9.3pt; }
th,td { border:1px solid #ccc; padding:4px 8px; text-align:left; }
th { background:#f0f0f0; }
td:nth-child(n+2){ text-align:right; }
hr { border:none; border-top:1px solid #ddd; margin:1.4em 0; }
.caption { font-size: 9pt; color:#444; text-align:center; font-style:italic;
  margin:.2em 0 1.2em; }
.mermaid { text-align:center; margin: .6em 0; }
blockquote { border-left:3px solid #ccc; margin:.6em 0; padding:.1em 1em; color:#333; }
a { color:#1a5fb4; text-decoration:none; }
h2, h3 { page-break-after: avoid; }
table, pre, .mermaid { page-break-inside: avoid; }
"""

doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Brush Optimization — Mkwizu</title>
<style>{CSS}</style>
<script>{MERMAID_JS}</script>
<script>
mermaid.initialize({{ startOnLoad: true, theme: 'neutral',
  flowchart: {{ htmlLabels: true, useMaxWidth: true }} }});
</script>
</head><body>
{body}
</body></html>"""

HTML.write_text(doc)
print(f"wrote {HTML} ({len(doc)} bytes, {len(mermaid_blocks)} mermaid figures)")
