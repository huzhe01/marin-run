#!/usr/bin/env python3
"""Render data.json into a self-contained dashboard.html."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime, timezone

# ---------------------------------------------------------------- helpers


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def human_int(value) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def human_tokens(value) -> str:
    if value is None:
        return "—"
    value = float(value)
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{unit}"
    return f"{value:.0f}"


def human_duration(seconds: float) -> str:
    if seconds is None:
        return "—"
    days, rem = divmod(int(seconds), 86400)
    hours = rem // 3600
    if days:
        return f"{days}d {hours}h"
    minutes = (rem % 3600) // 60
    return f"{hours}h {minutes}m"


# ---------------------------------------------------------------- charts


def line_chart(
    points: list[tuple[float, float]],
    *,
    width: int = 760,
    height: int = 200,
    stroke: str = "var(--accent)",
    fill: str = "var(--accent-wash)",
    y_label: str = "",
    log_y: bool = False,
    y_ticks: int = 4,
    clamp_zero: bool = False,
) -> str:
    """Inline SVG line chart with area fill, faint grid, emphasized endpoint."""
    if len(points) < 2:
        return '<p class="empty">not enough data yet</p>'

    pad_l, pad_r, pad_t, pad_b = 56, 14, 14, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    xs = [p[0] for p in points]
    raw_ys = [p[1] for p in points]
    if log_y:
        floor = min(y for y in raw_ys if y > 0) if any(y > 0 for y in raw_ys) else 1e-6
        ys = [math.log10(max(y, floor)) for y in raw_ys]
    else:
        ys = raw_ys

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        y_max = y_min + 1
    span = y_max - y_min
    y_min -= span * 0.08
    y_max += span * 0.08
    if clamp_zero and not log_y:
        y_min = max(0.0, y_min)

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min or 1) * plot_w

    def sy(y: float) -> float:
        return pad_t + (1 - (y - y_min) / (y_max - y_min)) * plot_h

    coords = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = (
        f"{path} L{coords[-1][0]:.1f},{pad_t + plot_h:.1f} "
        f"L{coords[0][0]:.1f},{pad_t + plot_h:.1f} Z"
    )

    grid, labels = [], []
    for i in range(y_ticks + 1):
        val = y_min + (y_max - y_min) * i / y_ticks
        y = sy(val)
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"/>'
        )
        shown = 10**val if log_y else val
        text = f"{shown:.3g}"
        labels.append(
            f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" text-anchor="end">{esc(text)}</text>'
        )

    x_labels = []
    for frac in (0, 0.5, 1):
        val = x_min + (x_max - x_min) * frac
        anchor = "start" if frac == 0 else ("end" if frac == 1 else "middle")
        x_labels.append(
            f'<text x="{sx(val):.1f}" y="{height - 8}" text-anchor="{anchor}">'
            f"{human_int(val)}</text>"
        )

    ex, ey = coords[-1]
    return f"""<svg class="chart" viewBox="0 0 {width} {height}" role="img"
 aria-label="{esc(y_label or 'chart')}" preserveAspectRatio="none">
  <g class="grid">{''.join(grid)}</g>
  <path class="area" d="{area}" fill="{fill}"/>
  <path class="line" d="{path}" stroke="{stroke}"/>
  <circle class="endpoint" cx="{ex:.1f}" cy="{ey:.1f}" r="3.5" fill="{stroke}"/>
  <g class="axis y">{''.join(labels)}</g>
  <g class="axis x">{''.join(x_labels)}</g>
</svg>"""


def bar_row(label: str, value: float, vmax: float, *, tone: str, note: str = "") -> str:
    pct = 0 if not vmax else max(0.0, min(1.0, value / vmax)) * 100
    return f"""<div class="bar-row">
  <span class="bar-label">{esc(label)}</span>
  <span class="bar-track"><span class="bar-fill t-{tone}" style="width:{pct:.2f}%"></span></span>
  <span class="bar-value">{value:.4f}</span>
  <span class="bar-note">{esc(note)}</span>
</div>"""


# ---------------------------------------------------------------- panels


def slice_table(summary: dict, family: str) -> str:
    """Per-slice bpb for one eval family.

    Runs that log both the routed and the dropless eval get a drop-cost column;
    the current hero run logs dropless only, so the column is omitted rather
    than showing an empty table.
    """
    if any(k.startswith(f"eval/{family}/") for k in summary):
        prefix, dropless_prefix = f"eval/{family}/", f"eval_dropless/{family}/"
    else:
        prefix, dropless_prefix = f"eval_dropless/{family}/", None
    rows = []
    for key, value in summary.items():
        if not (key.startswith(prefix) and key.endswith("/bpb")):
            continue
        name = key[len(prefix) : -len("/bpb")]
        if name in ("macro", "micro") or "/" not in key[len(prefix) :]:
            continue
        dropless = (
            summary.get(f"{dropless_prefix}{name}/bpb") if dropless_prefix else None
        )
        rows.append((name.replace("-llama3", ""), value, dropless))
    if not rows:
        return '<p class="empty">no slices reported yet</p>'
    rows.sort(key=lambda r: -r[1])
    vmax = max(r[1] for r in rows)

    out = []
    for name, routed, dropless in rows:
        cost = (routed - dropless) if dropless is not None else None
        tone = "neutral"
        if cost is not None:
            tone = "crit" if cost > 0.09 else ("warn" if cost > 0.06 else "ok")
        note = f"丢弃 +{cost:.3f}" if cost is not None else ""
        out.append(bar_row(name, routed, vmax, tone=tone, note=note))
    return "".join(out)


def router_strip(summary: dict) -> str:
    layers = []
    for key, value in summary.items():
        if key.startswith("train/router/layer_") and key.endswith(
            "/capacity_overflow_rate"
        ):
            idx = int(key.split("layer_")[1].split("/")[0])
            layers.append((idx, value))
    if not layers:
        return '<p class="empty">no router telemetry</p>'
    layers.sort()
    vmax = max(v for _, v in layers)
    cells = []
    for idx, value in layers:
        ratio = value / vmax if vmax else 0
        tone = "crit" if ratio > 0.85 else ("warn" if ratio > 0.6 else "ok")
        cells.append(
            f'<div class="rcell t-{tone}" style="--h:{ratio * 100:.1f}%" '
            f'title="layer {idx} · overflow {value * 100:.2f}%">'
            f'<span class="rbar"></span><span class="ridx">{idx}</span></div>'
        )
    return f'<div class="router-strip">{"".join(cells)}</div>'


def mixture_grid(summary: dict) -> str:
    """40 semantic clusters x 5 quality quintiles, live from the running job."""
    cells: dict[tuple[int, int], float] = {}
    for key, value in summary.items():
        if not key.startswith("mixture/weight/c"):
            continue
        token = key.rsplit("/", 1)[1]
        try:
            cluster = int(token[1 : token.index("q")])
            quality = int(token[token.index("q") + 1 :])
        except (ValueError, IndexError):
            continue
        cells[(cluster, quality)] = value
    if not cells:
        return '<p class="empty">no mixture telemetry</p>'

    vmax = max(cells.values())
    clusters = sorted({c for c, _ in cells})
    qualities = sorted({q for _, q in cells})

    head = "".join(f"<th>q{q}</th>" for q in qualities)
    body = []
    for cluster in clusters:
        row_total = sum(cells.get((cluster, q), 0) for q in qualities)
        tds = []
        for q in qualities:
            value = cells.get((cluster, q), 0.0)
            alpha = (value / vmax) ** 0.45 if vmax else 0
            tds.append(
                f'<td style="--a:{alpha:.3f}" title="c{cluster:02d} q{q} · '
                f'{value * 100:.3f}% of mix"></td>'
            )
        body.append(
            f"<tr><th>c{cluster:02d}</th>{''.join(tds)}"
            f'<td class="rowsum">{row_total * 100:.2f}%</td></tr>'
        )
    return f"""<div class="scroll-x"><table class="mixture">
<thead><tr><th></th>{head}<th class="rowsum">合计</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>"""


def status_timeline(entries: list[dict]) -> str:
    if not entries:
        return '<p class="empty">no log entries</p>'
    out = []
    for entry in reversed(entries):
        when = parse_ts(entry["created_at"])
        body = entry["body"].strip()
        agentic = body.startswith("🤖")
        body = body.lstrip("🤖").strip()
        # Keep the first substantive paragraph; these entries are long.
        para = next((p for p in body.split("\n\n") if p.strip()), body)
        para = " ".join(para.split())
        if len(para) > 460:
            para = para[:460].rsplit(" ", 1)[0] + "…"
        badge = '<span class="chip agent">agent</span>' if agentic else ""
        out.append(
            f"""<li class="event">
  <div class="event-meta">
    <time>{when:%m-%d %H:%M}</time>
    <span class="who">{esc(entry['author'])}</span>{badge}
  </div>
  <p class="event-body">{esc(para)}</p>
  <a class="event-link" href="{esc(entry['url'])}">原文 ↗</a>
</li>"""
        )
    return f'<ol class="events">{"".join(out)}</ol>'


def runs_table(runs: list[dict], now: datetime) -> str:
    rows = []
    for run in runs[:40]:
        created = parse_ts(run["created_at"])
        age = (now - created).total_seconds()
        state = run["state"]
        tone = {"running": "ok", "crashed": "crit", "finished": "neutral"}.get(
            state, "neutral"
        )
        loss = f"{run['loss']:.4f}" if isinstance(run["loss"], (int, float)) else "—"
        mfu = f"{run['mfu']:.1f}%" if isinstance(run["mfu"], (int, float)) else "—"
        rows.append(
            f"""<tr>
  <td class="mono name">{esc(run['name'][:52])}</td>
  <td><span class="chip t-{tone}">{esc(state)}</span></td>
  <td class="num">{human_int(run['step'])}</td>
  <td class="num">{loss}</td>
  <td class="num">{mfu}</td>
  <td class="num dim">{human_duration(age)} 前</td>
</tr>"""
        )
    return f"""<div class="scroll-x"><table class="runs">
<thead><tr><th>run</th><th>状态</th><th class="num">step</th>
<th class="num">loss</th><th class="num">MFU</th><th class="num">启动</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def issue_list(issues: list[dict], *, show_labels: bool = False) -> str:
    if not issues:
        return '<p class="empty">none open</p>'
    out = []
    for issue in issues:
        chips = ""
        if show_labels:
            keep = [
                l
                for l in issue.get("labels", [])
                if l not in ("experiment", "agent-generated")
            ]
            chips = "".join(f'<span class="chip">{esc(l)}</span>' for l in keep[:3])
        claimed = (
            '<span class="chip warn">已认领</span>'
            if issue.get("assigned")
            else ""
        )
        out.append(
            f"""<li>
  <a href="{esc(issue['url'])}"><span class="mono num-ref">#{issue['number']}</span>
  {esc(issue['title'])}</a>
  <span class="issue-meta">{issue['comments']} 评论{chips}{claimed}</span>
</li>"""
        )
    return f'<ul class="issues">{"".join(out)}</ul>'


# ---------------------------------------------------------------- page


def lineage_table(lineage: list[dict], current: str) -> str:
    """Every run id the campaign has trained under, oldest first."""
    if not lineage:
        return '<p class="empty">no lineage data</p>'
    rows = []
    for seg in lineage:
        is_current = seg["name"] == current
        tone = {"running": "ok", "crashed": "neutral", "finished": "neutral"}.get(
            seg["state"], "neutral"
        )
        state = "当前" if is_current else {"crashed": "已停止", "finished": "已结束"}.get(
            seg["state"], seg["state"]
        )
        rows.append(
            f"""<tr{' class="current"' if is_current else ''}>
  <td class="mono name">{esc(seg['name'])}</td>
  <td><span class="chip t-{'ok' if is_current else tone}">{esc(state)}</span></td>
  <td class="num">{parse_ts(seg['created_at']):%m-%d %H:%M}</td>
  <td class="num">{parse_ts(seg['heartbeat_at']):%m-%d %H:%M}</td>
  <td class="num">{human_int(seg['step'])}</td>
</tr>"""
        )
    return f"""<div class="scroll-x"><table class="runs lineage">
<thead><tr><th>run id</th><th>状态</th><th class="num">启动 (UTC)</th>
<th class="num">最后心跳</th><th class="num">最后 step</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


EVENT_LIMIT = 30


def build(data: dict) -> str:
    hero = data["hero"]
    summary = hero["summary"]
    config = hero["config"]
    gh_data = data["github"]
    now = datetime.now(timezone.utc)

    step = summary.get("global_step") or summary.get("_step") or 0
    total_steps = config.get("trainer.trainer.num_train_steps") or config.get(
        "stop_after_steps"
    )
    progress = summary.get("run_progress") or (step / total_steps if total_steps else 0)
    tokens_done = summary.get("throughput/total_tokens")
    batch = config.get("trainer.trainer.train_batch_size")
    seq = config.get("model.max_seq_len")
    tokens_target = (
        total_steps * batch * seq if (total_steps and batch and seq) else None
    )

    # The run is relaunched under a new id after interventions, but run_progress
    # counts from step 0 of the campaign. Measuring wall time from the current
    # segment's start divides a short duration by the cumulative progress and
    # collapses the ETA (a 69-day remainder read as 7), so use the first segment.
    created = parse_ts(hero["created_at"])
    lineage = data.get("lineage") or []
    started = parse_ts(lineage[0]["created_at"]) if lineage else created
    heartbeat = parse_ts(hero["heartbeat_at"])
    wall = (now - started).total_seconds()
    seg_wall = (now - created).total_seconds()
    active = summary.get("_runtime") or seg_wall
    availability = active / seg_wall if seg_wall else 1.0
    hb_age = (now - heartbeat).total_seconds()

    eta_days = (wall / progress - wall) / 86400 if progress else None
    tps = summary.get("throughput/tokens_per_second")
    compute_days = (
        (tokens_target - tokens_done) / tps / 86400
        if (tokens_target and tokens_done and tps)
        else None
    )

    live = hero["state"] == "running" and hb_age < 3600
    state_tone = "ok" if live else ("warn" if hero["state"] == "running" else "crit")
    state_text = (
        "训练中" if live else ("心跳停滞" if hero["state"] == "running" else hero["state"])
    )

    loss_pts = [
        (p["global_step"], p["train/loss"])
        for p in hero["loss_history"]
        if p.get("global_step") is not None and p.get("train/loss") is not None
    ]
    mfu_pts = [
        (p["global_step"], p["throughput/mfu"])
        for p in hero["health_history"]
        if p.get("global_step") is not None and p.get("throughput/mfu") is not None
    ]
    drop_pts = [
        (p["global_step"], p["moe/drop_fraction"] * 100)
        for p in hero["health_history"]
        if p.get("global_step") is not None and p.get("moe/drop_fraction") is not None
    ]
    # Skip the step-0 eval: it measures random init and squashes the rest of the axis.
    eval_key = (
        "eval/bpb"
        if any(p.get("eval/bpb") is not None for p in hero["eval_history"])
        else "eval_dropless/bpb"
    )
    slice_note = (
        "条长 = bpb，颜色 = 丢弃代价"
        if any(k.startswith("eval/paloma/") for k in summary)
        else "条长 = bpb（本次跑只记无丢弃口径）"
    )
    eval_pts = [
        (p["global_step"], p[eval_key])
        for p in hero["eval_history"]
        if p.get("global_step") and p.get(eval_key) is not None
    ]

    kpis = [
        ("train/loss", f"{summary.get('train/loss', 0):.4f}", "最近一步"),
        ("MFU", f"{summary.get('throughput/mfu', 0):.2f}%", "GB200 理论峰值占比"),
        ("吞吐", f"{(tps or 0) / 1e6:.2f}M", "tokens / 秒"),
        (
            "路由丢弃",
            f"{summary.get('moe/drop_fraction', 0) * 100:.2f}%",
            f"发送 {summary.get('moe/sender_drop_fraction', 0) * 100:.2f}% · "
            f"接收 {summary.get('moe/receiver_drop_fraction', 0) * 100:.3f}%",
        ),
        (
            "eval bpb",
            f"{(summary.get('eval/bpb') or summary.get('eval_dropless/bpb') or 0):.4f}",
            "无丢弃口径"
            if summary.get("eval/bpb") is None
            else f"无丢弃 {summary.get('eval_dropless/bpb', 0):.4f}",
        ),
        (
            "本段可用率",
            f"{availability * 100:.1f}%",
            f"活跃 {human_duration(active)} / 本段挂钟 {human_duration(seg_wall)}",
        ),
    ]
    kpi_html = "".join(
        f"""<div class="kpi">
  <span class="kpi-label">{esc(label)}</span>
  <span class="kpi-value">{esc(value)}</span>
  <span class="kpi-note">{esc(note)}</span>
</div>"""
        for label, value, note in kpis
    )

    spec = [
        ("总参数", f"{summary.get('parameter_count', 0) / 1e9:.1f}B"),
        (
            "激活参数",
            f"{config.get('model.num_experts_per_token', 0)}/{config.get('model.num_experts', 0)} 专家",
        ),
        (
            "结构",
            f"d{config.get('model.hidden_dim')} · {config.get('model.num_layers')}L · "
            f"{config.get('model.num_heads')}H",
        ),
        ("序列", f"{config.get('model.max_seq_len')} · 滑窗 {config.get('model.sliding_window', '—')}"),
        ("设备", f"{human_int(summary.get('num_devices'))} × {summary.get('throughput/device_kind', '—')}"),
        (
            "并行",
            f"EP{config.get('trainer.expert_axis_size')} × "
            f"{config.get('trainer.replica_axis_size')} 机架",
        ),
        ("批大小", f"{human_int(batch)} 序列 · {human_tokens((batch or 0) * (seq or 0))} tokens/步"),
        ("容量因子", f"{config.get('model.capacity_factor')} · {config.get('model.num_expert_waves')} wave"),
    ]
    spec_html = "".join(
        f'<div class="spec-item"><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>'
        for k, v in spec
    )

    gates = gh_data["burndown"]
    done = sum(1 for g in gates if g["done"])
    gate_html = "".join(
        f"""<li class="gate {'done' if g['done'] else 'open'}">
  <span class="gate-mark">{'✓' if g['done'] else '○'}</span>
  <span>{esc(g['text'][:150])}</span></li>"""
        for g in gates
    )

    unassigned = [i for i in gh_data["experiments"] if not i["assigned"]]
    commits = gh_data["commits"][:14]
    commit_html = "".join(
        f"""<li>
  <a href="{esc(c['url'])}"><code>{esc(c['sha'])}</code> {esc(c['message'][:96])}</a>
  <span class="issue-meta">{esc(c['author'])} · {parse_ts(c['date']):%m-%d %H:%M}</span>
</li>"""
        for c in commits
    )

    collected = parse_ts(data["collected_at"])

    return f"""<title>Marin Hero Run 观测台</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;1,400&display=swap">
<style>
:root {{
  --ground: #f4f5f8;
  --panel: #ffffff;
  --panel-2: #eef0f5;
  --line: #d8dce6;
  --line-soft: #e7eaf1;
  --ink: #171a21;
  --ink-2: #4a5163;
  --ink-3: #767e94;
  --accent: #4a52c4;
  --accent-wash: rgba(74, 82, 196, 0.12);
  --ok: #2f7a52;
  --warn: #9c6a13;
  --crit: #b5352f;
  --ok-wash: rgba(47, 122, 82, 0.14);
  --warn-wash: rgba(156, 106, 19, 0.16);
  --crit-wash: rgba(181, 53, 47, 0.14);
  --sans: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --serif: "IBM Plex Serif", Georgia, "Songti SC", serif;
  --r: 4px;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #0f1218;
    --panel: #161a23;
    --panel-2: #1c2130;
    --line: #2b3242;
    --line-soft: #232936;
    --ink: #e6e9f0;
    --ink-2: #a8b0c2;
    --ink-3: #737c92;
    --accent: #8f96ee;
    --accent-wash: rgba(143, 150, 238, 0.16);
    --ok: #5cb684;
    --warn: #d3a248;
    --crit: #e0706a;
    --ok-wash: rgba(92, 182, 132, 0.16);
    --warn-wash: rgba(211, 162, 72, 0.18);
    --crit-wash: rgba(224, 112, 106, 0.16);
  }}
}}
:root[data-theme="dark"] {{
  --ground: #0f1218;
  --panel: #161a23;
  --panel-2: #1c2130;
  --line: #2b3242;
  --line-soft: #232936;
  --ink: #e6e9f0;
  --ink-2: #a8b0c2;
  --ink-3: #737c92;
  --accent: #8f96ee;
  --accent-wash: rgba(143, 150, 238, 0.16);
  --ok: #5cb684;
  --warn: #d3a248;
  --crit: #e0706a;
  --ok-wash: rgba(92, 182, 132, 0.16);
  --warn-wash: rgba(211, 162, 72, 0.18);
  --crit-wash: rgba(224, 112, 106, 0.16);
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1160px; margin: 0 auto; padding: 40px 24px 72px; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
a:focus-visible, [tabindex]:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.mono, code {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
.num {{ text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }}
.dim {{ color: var(--ink-3); }}
.empty {{ color: var(--ink-3); font-style: italic; margin: 0; }}
.scroll-x {{ overflow-x: auto; }}

/* ---- masthead ---- */
.mast {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 14px; margin-bottom: 6px; }}
.mast h1 {{
  font-size: 27px; font-weight: 600; letter-spacing: -0.02em;
  margin: 0; text-wrap: balance;
}}
.mast .run-id {{ font-family: var(--mono); font-size: 13px; color: var(--ink-3); }}
.subline {{ color: var(--ink-2); font-size: 14px; margin: 0 0 26px; max-width: 74ch; }}

.chip {{
  display: inline-flex; align-items: center; gap: 5px;
  font-family: var(--mono); font-size: 11px; font-weight: 500;
  letter-spacing: 0.04em; text-transform: uppercase;
  padding: 2px 7px; border-radius: var(--r);
  background: var(--panel-2); color: var(--ink-2); border: 1px solid var(--line-soft);
  white-space: nowrap;
}}
.chip.t-ok, .chip.ok {{ background: var(--ok-wash); color: var(--ok); border-color: transparent; }}
.chip.t-warn, .chip.warn {{ background: var(--warn-wash); color: var(--warn); border-color: transparent; }}
.chip.t-crit, .chip.crit {{ background: var(--crit-wash); color: var(--crit); border-color: transparent; }}
.chip.agent {{ background: var(--accent-wash); color: var(--accent); border-color: transparent; }}
.chip.live::before {{
  content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: currentColor; animation: pulse 2.4s ease-in-out infinite;
}}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.25; }} }}
@media (prefers-reduced-motion: reduce) {{ .chip.live::before {{ animation: none; }} }}

/* ---- progress ---- */
.progress-card {{
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--r); padding: 22px 24px; margin-bottom: 18px;
}}
.progress-head {{
  display: flex; flex-wrap: wrap; justify-content: space-between;
  align-items: flex-end; gap: 16px; margin-bottom: 14px;
}}
.progress-pct {{
  font-family: var(--mono); font-size: 46px; font-weight: 600;
  line-height: 1; letter-spacing: -0.03em;
}}
.progress-pct small {{ font-size: 20px; color: var(--ink-3); font-weight: 400; }}
.progress-facts {{ display: flex; flex-wrap: wrap; gap: 26px; }}
.progress-facts div {{ display: flex; flex-direction: column; gap: 2px; }}
.progress-facts dt, .fact-label {{
  font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-3);
}}
.progress-facts dd, .fact-value {{
  margin: 0; font-family: var(--mono); font-size: 15px; font-variant-numeric: tabular-nums;
}}
.track {{
  height: 10px; background: var(--panel-2); border-radius: 2px;
  overflow: hidden; border: 1px solid var(--line-soft);
}}
.track span {{ display: block; height: 100%; background: var(--accent); }}
.track-legend {{
  display: flex; justify-content: space-between;
  font-family: var(--mono); font-size: 11px; color: var(--ink-3); margin-top: 6px;
}}

/* ---- grids ---- */
.kpis {{
  display: grid; grid-template-columns: repeat(6, 1fr);
  gap: 1px; background: var(--line); border: 1px solid var(--line);
  border-radius: var(--r); overflow: hidden; margin-bottom: 34px;
}}
@media (max-width: 980px) {{ .kpis {{ grid-template-columns: repeat(3, 1fr); }} }}
@media (max-width: 560px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} }}
.kpi {{ background: var(--panel); padding: 14px 16px; display: flex; flex-direction: column; gap: 3px; }}
.kpi-label {{ font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-3); }}
.kpi-value {{ font-family: var(--mono); font-size: 23px; font-weight: 600; letter-spacing: -0.01em; }}
.kpi-note {{ font-size: 11.5px; color: var(--ink-3); font-family: var(--mono); }}

section {{ margin-bottom: 40px; }}
.sec-head {{
  display: flex; align-items: baseline; gap: 12px;
  border-bottom: 1px solid var(--line); padding-bottom: 8px; margin-bottom: 18px;
}}
.sec-head h2 {{ font-size: 17px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }}
.sec-head p {{ margin: 0; font-size: 13px; color: var(--ink-3); }}

.cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 18px; }}
.card {{
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--r); padding: 18px 20px;
}}
.card h3 {{
  font-size: 12px; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--ink-3); margin: 0 0 12px; font-weight: 600;
}}
.h3-note {{ text-transform: none; letter-spacing: 0; font-weight: 400; opacity: 0.85; }}

/* ---- charts ---- */
.chart {{ width: 100%; height: auto; display: block; }}
.chart .grid line {{ stroke: var(--line-soft); stroke-width: 1; }}
.chart .line {{ fill: none; stroke-width: 1.75; stroke-linejoin: round; stroke-linecap: round; }}
.chart .axis text {{
  font-family: var(--mono); font-size: 10px; fill: var(--ink-3);
}}

/* ---- bars ---- */
.bar-row {{
  display: grid; grid-template-columns: minmax(96px, 1.3fr) minmax(56px, 2.2fr) 58px 82px;
  align-items: center; gap: 9px; padding: 3px 0;
  font-size: 12.5px;
}}
.bar-label {{ font-family: var(--mono); color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bar-track {{ height: 7px; background: var(--panel-2); border-radius: 2px; overflow: hidden; }}
.bar-fill {{ display: block; height: 100%; background: var(--accent); }}
.bar-fill.t-ok {{ background: var(--ok); }}
.bar-fill.t-warn {{ background: var(--warn); }}
.bar-fill.t-crit {{ background: var(--crit); }}
.bar-value {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }}
.bar-note {{ font-family: var(--mono); font-size: 11px; color: var(--ink-3); white-space: nowrap; }}

/* ---- router ---- */
.router-strip {{ display: flex; align-items: flex-end; gap: 2px; height: 118px; }}
.rcell {{ flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%; gap: 3px; }}
.rbar {{ width: 100%; height: var(--h); background: var(--ok); border-radius: 1px 1px 0 0; min-height: 2px; }}
.rcell.t-warn .rbar {{ background: var(--warn); }}
.rcell.t-crit .rbar {{ background: var(--crit); }}
.ridx {{ font-family: var(--mono); font-size: 8px; color: var(--ink-3); }}
.rcell:nth-child(even) .ridx {{ visibility: hidden; }}

/* ---- mixture ---- */
table {{ border-collapse: collapse; width: 100%; }}
.mixture {{ font-family: var(--mono); font-size: 10.5px; }}
.mixture th {{ color: var(--ink-3); font-weight: 500; padding: 2px 4px; text-align: left; }}
.mixture thead th {{ text-align: center; }}
.mixture td {{
  height: 15px; min-width: 30px; padding: 0;
  background: color-mix(in srgb, var(--accent) calc(var(--a, 0) * 100%), var(--panel-2));
  border: 1px solid var(--panel);
}}
.mixture td.rowsum {{
  background: none; border: none; color: var(--ink-3);
  text-align: right; padding-left: 8px; font-variant-numeric: tabular-nums;
}}
.mixture th.rowsum {{ text-align: right; }}

/* ---- events ---- */
.events {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; }}
.event {{
  border-left: 2px solid var(--line); padding: 0 0 20px 18px; position: relative;
}}
.event::before {{
  content: ""; position: absolute; left: -5px; top: 6px;
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--panel); border: 2px solid var(--accent);
}}
.event:last-child {{ border-left-color: transparent; padding-bottom: 0; }}
.event-meta {{ display: flex; align-items: center; gap: 9px; margin-bottom: 4px; }}
.event-meta time {{ font-family: var(--mono); font-size: 12px; color: var(--ink-3); }}
.event-meta .who {{ font-family: var(--mono); font-size: 12px; color: var(--ink-2); }}
.event-body {{
  font-family: var(--serif); font-size: 14.5px; line-height: 1.6;
  margin: 0 0 5px; color: var(--ink-2); max-width: 76ch;
}}
.event-link {{ font-size: 12px; font-family: var(--mono); }}

/* ---- tables & lists ---- */
.runs {{ font-size: 12.5px; }}
.runs th {{
  text-align: left; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink-3); font-weight: 600; padding: 6px 10px; border-bottom: 1px solid var(--line);
}}
.runs td {{ padding: 5px 10px; border-bottom: 1px solid var(--line-soft); }}
.runs td.name {{ color: var(--ink-2); }}
.runs tbody tr:hover {{ background: var(--panel-2); }}

.issues, .gates, .commits {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 9px; }}
.issues li, .commits li {{ display: flex; flex-direction: column; gap: 2px; font-size: 13.5px; }}
.num-ref {{ color: var(--ink-3); font-size: 12px; }}
.issue-meta {{ display: flex; align-items: center; gap: 7px; font-size: 11.5px; color: var(--ink-3); font-family: var(--mono); }}
.gates {{ gap: 5px; }}
.gate {{ display: flex; gap: 9px; font-size: 13px; align-items: baseline; }}
.gate-mark {{ font-family: var(--mono); }}
.gate.done {{ color: var(--ink-3); }}
.gate.done .gate-mark {{ color: var(--ok); }}
.gate.open .gate-mark {{ color: var(--warn); }}
.gate.open {{ color: var(--ink); }}

.spec {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: var(--r); overflow: hidden; margin: 0; }}
.spec-item {{ background: var(--panel); padding: 11px 15px; }}
.spec-item dt {{ font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); }}
.spec-item dd {{ margin: 2px 0 0; font-family: var(--mono); font-size: 13.5px; }}

.nav {{
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 18px;
  font-size: 13px; margin-bottom: 22px; padding-bottom: 12px;
  border-bottom: 1px solid var(--line-soft);
}}
.nav a {{ color: var(--ink-2); }}
.nav a[aria-current="page"] {{ color: var(--ink); font-weight: 600; }}
.nav-note {{
  margin-left: auto; font-family: var(--mono); font-size: 11px;
  letter-spacing: 0.05em; color: var(--ink-3);
}}
.lineage tr.current td {{ background: var(--accent-wash); }}
.more {{ margin: 12px 0 0; font-size: 13px; }}

footer {{
  border-top: 1px solid var(--line); padding-top: 18px; margin-top: 46px;
  font-size: 12.5px; color: var(--ink-3); display: flex; flex-wrap: wrap; gap: 18px;
}}
footer code {{ background: var(--panel-2); padding: 1px 5px; border-radius: 3px; }}
</style>

<div class="wrap">
  <nav class="nav" aria-label="站点">
    <a href="./" aria-current="page">实时看板</a>
    <a href="./pipeline.html">全流程拆解</a>
    <a href="./reports/">更新报告</a>
    <span class="nav-note">非官方 · 社区自建</span>
  </nav>
  <div class="mast">
    <h1>Marin Hero Run 观测台</h1>
    <span class="chip t-{state_tone} {'live' if live else ''}">{esc(state_text)}</span>
    <span class="run-id">{esc(hero['display_name'])}</span>
  </div>
  <p class="subline">
    535B-A23B MoE 在 704 张 GB200 上的全程公开预训练。全部数据来自
    <a href="https://wandb.ai/{esc('marin-community')}/marin_moe">W&amp;B 公开项目</a>
    与 <a href="{esc(gh_data['issue_urls']['status'])}">GitHub 事件日志</a>，无需任何凭证。
    这是基于公开数据的社区自建页面，不是 Marin 官方发布。
    快照时间 {collected:%Y-%m-%d %H:%M} UTC，最后心跳 {human_duration(hb_age)} 前。
  </p>

  <div class="progress-card">
    <div class="progress-head">
      <div class="progress-pct">{progress * 100:.2f}<small>%</small></div>
      <div class="progress-facts">
        <div><dt>step</dt><dd>{human_int(step)} / {human_int(total_steps)}</dd></div>
        <div><dt>tokens</dt><dd>{human_tokens(tokens_done)} / {human_tokens(tokens_target)}</dd></div>
        <div><dt>已运行</dt><dd>{human_duration(wall)}</dd></div>
        <div><dt>按挂钟外推剩余</dt><dd>{f'{eta_days:.0f} 天' if eta_days else '—'}</dd></div>
        <div><dt>按当前算力剩余</dt><dd>{f'{compute_days:.0f} 天' if compute_days else '—'}</dd></div>
      </div>
    </div>
    <div class="track"><span style="width:{progress * 100:.3f}%"></span></div>
    <div class="track-legend">
      <span>{started:%Y-%m-%d} 开跑 · 当前段自 {created:%m-%d %H:%M} UTC</span>
      <span>两条 ETA 的差值 = 故障与重启的代价</span>
    </div>
  </div>

  <div class="kpis">{kpi_html}</div>

  <section>
    <div class="sec-head">
      <h2>主链接力</h2>
      <p>每次干预后在新 run id 下续跑；按相同步数预算识别，含已放弃的试验段</p>
    </div>
    {lineage_table(lineage, hero['display_name'])}
  </section>

  <section>
    <div class="sec-head"><h2>训练曲线</h2><p>全程采样</p></div>
    <div class="cols">
      <div class="card">
        <h3>train/loss</h3>
        {line_chart(loss_pts, y_label="training loss", log_y=True)}
      </div>
      <div class="card">
        <h3>MFU（%）</h3>
        {line_chart(mfu_pts, y_label="MFU", stroke="var(--ok)", fill="var(--ok-wash)")}
      </div>
      <div class="card">
        <h3>路由丢弃率（%）</h3>
        {line_chart(drop_pts, y_label="drop fraction", stroke="var(--warn)",
                    fill="var(--warn-wash)", clamp_zero=True)}
      </div>
      <div class="card">
        <h3>eval bpb <span class="h3-note">（{esc(eval_key)}，已跳过 step 0）</span></h3>
        {line_chart(eval_pts, y_label="eval bpb", stroke="var(--accent)", fill="var(--accent-wash)")}
      </div>
    </div>
  </section>

  <section>
    <div class="sec-head">
      <h2>评测分片</h2>
      <p>bits per byte，越低越好；右列是路由丢弃相对无丢弃基线的代价</p>
    </div>
    <div class="cols">
      <div class="card">
        <h3>Paloma <span class="h3-note">{slice_note}</span></h3>
        {slice_table(summary, "paloma")}
      </div>
      <div class="card">
        <h3>Uncheatable Eval <span class="h3-note">{slice_note}</span></h3>
        {slice_table(summary, "uncheatable_eval")}
      </div>
    </div>
  </section>

  <section>
    <div class="sec-head">
      <h2>MoE 路由健康</h2>
      <p>每层的容量溢出率，48 层</p>
    </div>
    <div class="card">{router_strip(summary)}</div>
  </section>

  <section>
    <div class="sec-head">
      <h2>实时数据配比</h2>
      <p>40 个语义簇 × 5 个质量分位，取自运行中作业的 mixture 权重</p>
    </div>
    <div class="card">{mixture_grid(summary)}</div>
  </section>

  <section>
    <div class="sec-head"><h2>模型与拓扑</h2></div>
    <dl class="spec">{spec_html}</dl>
  </section>

  <section>
    <div class="sec-head">
      <h2>事故与干预日志</h2>
      <p>来自 issue #8506，含 agent 自动记录</p>
    </div>
    {status_timeline(gh_data["status_log"][-EVENT_LIMIT:])}
    {(f'<p class="more"><a href="{esc(gh_data["issue_urls"]["status"])}">'
      f'在 GitHub 上查看全部 {len(gh_data["status_log"])} 条 ↗</a></p>')
     if len(gh_data["status_log"]) > EVENT_LIMIT else ''}
  </section>

  <section>
    <div class="sec-head">
      <h2>周边探索</h2>
      <p>marin_moe 项目最近的消融与诊断跑</p>
    </div>
    {runs_table(data["recent_runs"], now)}
  </section>

  <section>
    <div class="sec-head">
      <h2>启动门禁</h2>
      <p>{done} / {len(gates)} 已关闭</p>
    </div>
    <div class="card"><ul class="gates">{gate_html}</ul></div>
  </section>

  <section>
    <div class="sec-head"><h2>可参与的口子</h2></div>
    <div class="cols">
      <div class="card">
        <h3>help-wanted（全仓库仅 {len(gh_data['help_wanted'])} 个）</h3>
        {issue_list(gh_data["help_wanted"])}
      </div>
      <div class="card">
        <h3>无人认领的实验 issue（{len(unassigned)}）</h3>
        {issue_list(unassigned[:12], show_labels=True)}
      </div>
    </div>
  </section>

  <section>
    <div class="sec-head"><h2>最近提交</h2></div>
    <div class="card"><ul class="commits">{commit_html}</ul></div>
  </section>

  <footer>
    <span>每 6 小时由 GitHub Actions 自动刷新 · <a href="https://github.com/huzhe01/marin-run">源码</a></span>
    <span><a href="{esc(gh_data['issue_urls']['spec'])}">模型规格 #8435</a></span>
    <span><a href="{esc(gh_data['issue_urls']['burndown'])}">门禁 #8233</a></span>
    <span><a href="https://marin.community/">marin.community</a></span>
  </footer>
</div>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.json")
    parser.add_argument("--out", default="dashboard.html")
    args = parser.parse_args()

    with open(args.data) as fh:
        data = json.load(fh)
    with open(args.out, "w") as fh:
        fh.write(build(data))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
