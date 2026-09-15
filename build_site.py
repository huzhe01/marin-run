#!/usr/bin/env python3
"""Assemble the static site from data.json, the explainer page and the reports.

    python3 collect.py --out data.json
    python3 build_site.py --data data.json --out site

Writes:
  site/index.html            live dashboard (render.py)
  site/pipeline.html         the pipeline explainer, with a link back
  site/reports/index.html    report list, newest first
  site/reports/<date>.html   one page per reports/<date>.md
  data/history.csv           one row appended per new step (tracked in git)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import markdown

import render

ROOT = Path(__file__).resolve().parent
HISTORY_FIELDS = [
    "collected_at",
    "run",
    "global_step",
    "run_progress",
    "train_loss",
    "eval_bpb",
    "mfu",
    "tokens_per_second",
    "total_tokens",
    "drop_fraction",
    "grad_norm",
]

REPORT_CSS = """
:root {
  --ground: #f4f5f8; --panel: #ffffff; --line: #d8dce6; --line-soft: #e7eaf1;
  --ink: #171a21; --ink-2: #4a5163; --ink-3: #767e94; --accent: #4a52c4;
  --sans: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "PingFang SC", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0f1218; --panel: #161a23; --line: #2b3242; --line-soft: #232936;
    --ink: #e6e9f0; --ink-2: #a8b0c2; --ink-3: #737c92; --accent: #8f96ee;
  }
}
:root[data-theme="dark"] {
  --ground: #0f1218; --panel: #161a23; --line: #2b3242; --line-soft: #232936;
  --ink: #e6e9f0; --ink-2: #a8b0c2; --ink-3: #737c92; --accent: #8f96ee;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink);
  font-family: var(--sans); font-size: 15.5px; line-height: 1.7; }
.wrap { max-width: 860px; margin: 0 auto; padding: 40px 24px 80px; }
a { color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.nav { display: flex; flex-wrap: wrap; align-items: baseline; gap: 18px; font-size: 13px;
  margin-bottom: 26px; padding-bottom: 12px; border-bottom: 1px solid var(--line-soft); }
.nav a { color: var(--ink-2); text-decoration: none; }
.nav a[aria-current="page"] { color: var(--ink); font-weight: 600; }
.nav-note { margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
h1 { font-size: 28px; line-height: 1.25; letter-spacing: -0.02em; margin: 0 0 6px; text-wrap: balance; }
h2 { font-size: 19px; margin: 38px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); text-wrap: balance; }
h3 { font-size: 15px; margin: 26px 0 8px; }
.meta { font-family: var(--mono); font-size: 12.5px; color: var(--ink-3); margin: 0 0 24px; }
.disclaimer { font-size: 13px; color: var(--ink-3); border-left: 3px solid var(--line);
  padding: 4px 0 4px 12px; margin: 0 0 28px; }
p, li { max-width: 72ch; }
code { font-family: var(--mono); font-size: 0.88em; background: var(--panel);
  border: 1px solid var(--line-soft); padding: 1px 5px; border-radius: 3px; }
pre { background: var(--panel); border: 1px solid var(--line); padding: 12px 14px;
  overflow-x: auto; border-radius: 4px; }
pre code { border: none; padding: 0; background: none; }
.table-wrap { overflow-x: auto; margin: 14px 0 18px; }
table { border-collapse: collapse; font-size: 13.5px; min-width: 420px; }
th, td { padding: 6px 12px 6px 0; border-bottom: 1px solid var(--line-soft); text-align: left; vertical-align: top; }
th { font-size: 11.5px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-3);
  border-bottom: 1px solid var(--line); }
td { font-variant-numeric: tabular-nums; }
blockquote { margin: 14px 0; padding: 2px 14px; border-left: 3px solid var(--accent); color: var(--ink-2); }
.reports { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 14px; }
.reports li { background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 14px 16px; }
.reports time { font-family: var(--mono); font-size: 12.5px; color: var(--ink-3); display: block; }
.reports a { font-weight: 600; text-decoration: none; }
"""

FONTS = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family='
    "IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600;700&display=swap\">"
)


def wrap_fragment(fragment: str) -> str:
    """Turn an artifact-style fragment (<title>/<link>/<style> then body markup)
    into a complete HTML document with those elements in <head>."""
    cut = fragment.index("</style>") + len("</style>")
    head, body = fragment[:cut], fragment[cut:]
    return (
        '<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def nav(prefix: str, current: str) -> str:
    items = [("", "实时看板"), ("pipeline.html", "全流程拆解"), ("reports/", "更新报告")]
    current_attr = ' aria-current="page"'
    links = "".join(
        f'<a href="{prefix}{href}"{current_attr if href == current else ""}>{label}</a>'
        for href, label in items
    )
    return (
        f'<nav class="nav" aria-label="站点">{links}'
        '<span class="nav-note">非官方 · 社区自建</span></nav>'
    )


def page(title: str, body: str, prefix: str, current: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
{FONTS}
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="wrap">
{nav(prefix, current)}
{body}
</div>
</body>
</html>
"""


def parse_report(path: Path) -> tuple[str, str, str]:
    """Return (title, date, html body) for a reports/<date>.md file.

    The first ``# `` heading is the title; everything after it is the body.
    """
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^# (.+)$", text, flags=re.M)
    title = match.group(1).strip() if match else path.stem
    body_md = text[match.end():] if match else text
    body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace(
        "</table>", "</table></div>"
    )
    return title, path.stem, body


def build_reports(out: Path) -> int:
    reports_dir = out / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for md in sorted((ROOT / "reports").glob("*.md"), reverse=True):
        title, date, body = parse_report(md)
        doc = f"<h1>{html.escape(title)}</h1>\n{body}"
        (reports_dir / f"{date}.html").write_text(
            page(title, doc, "../", "reports/"), encoding="utf-8"
        )
        entries.append((date, title))
    items = "".join(
        f'<li><time>{date}</time><a href="./{date}.html">{html.escape(title)}</a></li>'
        for date, title in entries
    )
    index = (
        "<h1>更新报告</h1>"
        '<p class="meta">基于公开的 GitHub issue、W&amp;B 与仓库提交整理，按时间倒序。</p>'
        f'<ul class="reports">{items or "<li>暂无报告</li>"}</ul>'
    )
    (reports_dir / "index.html").write_text(
        page("Marin 更新报告", index, "../", "reports/"), encoding="utf-8"
    )
    return len(entries)


def build_pipeline(out: Path) -> None:
    src = (ROOT / "pages" / "pipeline.html").read_text(encoding="utf-8")
    back = (
        '<p style="font-size:13px;margin:0 0 20px">'
        '<a href="./">← 实时看板</a> · <a href="./reports/">更新报告</a> · '
        '<span style="opacity:.7">非官方 · 社区自建</span></p>'
    )
    src = src.replace('<div class="page">', f'<div class="page">\n{back}', 1)
    (out / "pipeline.html").write_text(wrap_fragment(src), encoding="utf-8")


def append_history(data: dict, path: Path) -> bool:
    """Append one row when the run has moved since the last recorded row."""
    s = data["hero"]["summary"]
    row = {
        "collected_at": data["collected_at"],
        "run": data["hero"]["display_name"],
        "global_step": s.get("global_step"),
        "run_progress": s.get("run_progress"),
        "train_loss": s.get("train/loss"),
        "eval_bpb": s.get("eval/bpb") or s.get("eval_dropless/bpb"),
        "mfu": s.get("throughput/mfu"),
        "tokens_per_second": s.get("throughput/tokens_per_second"),
        "total_tokens": s.get("throughput/total_tokens"),
        "drop_fraction": s.get("moe/drop_fraction"),
        "grad_norm": s.get("grad/norm/total"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows and rows[-1]["run"] == row["run"] and rows[-1]["global_step"] == str(row["global_step"]):
            return False
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.json")
    parser.add_argument("--out", default="site")
    parser.add_argument("--history", default="data/history.csv")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    (out / "index.html").write_text(wrap_fragment(render.build(data)), encoding="utf-8")
    build_pipeline(out)
    n_reports = build_reports(out)
    (out / ".nojekyll").write_text("")
    appended = append_history(data, ROOT / args.history)
    print(
        f"site -> {out}/ (dashboard, pipeline, {n_reports} reports); "
        f"history {'appended' if appended else 'unchanged'} "
        f"at {datetime.now():%Y-%m-%d %H:%M}"
    )


if __name__ == "__main__":
    main()
