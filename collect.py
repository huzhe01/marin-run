#!/usr/bin/env python3
"""Collect public Marin hero-run telemetry into data.json.

Sources (all public, no credentials needed):
  - W&B GraphQL  : live run state, loss history, eval slices, MoE health, data mixture
  - GitHub API   : hero-run incident log, launch burndown, recent commits/experiments

Usage:  python3 collect.py [--out data.json] [--run <wandb run id>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

WANDB_GQL = "https://api.wandb.ai/graphql"
ENTITY = "marin-community"
PROJECT = "marin_moe"
REPO = "marin-community/marin"

# GitHub issues that carry the run narrative.
STATUS_ISSUE = 8506   # Hero Run / Ongoing Status  -- incident + intervention log
SPEC_ISSUE = 8435     # [Hero Run] 535B-A23B on 18T tokens -- model + data spec
BURNDOWN_ISSUE = 8233  # Next hero run burndown -- launch gates


def gql(query: str, variables: dict | None = None, attempts: int = 4) -> dict:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    for attempt in range(attempts):
        req = urllib.request.Request(
            WANDB_GQL, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            # The W&B endpoint drops long connections intermittently.
            if attempt == attempts - 1:
                raise
            wait = 2**attempt
            print(f"  ! wandb {exc}; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if "errors" in body:
            raise RuntimeError(f"wandb: {json.dumps(body['errors'])[:400]}")
        return body["data"]
    raise RuntimeError("unreachable")


def gh(path: str) -> object:
    """GitHub read via the gh CLI, falling back to anonymous HTTP."""
    try:
        out = subprocess.run(
            ["gh", "api", path], capture_output=True, text=True, timeout=90
        )
        if out.returncode == 0:
            return json.loads(out.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    try:
        req = urllib.request.Request(
            f"https://api.github.com/{path.lstrip('/')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        print(f"  ! github {path}: {exc}", file=sys.stderr)
        return None


def gh_all(path: str, per_page: int = 100) -> list:
    """Every page of a GitHub list endpoint.

    Issue comments come back oldest first, so a single page silently drops the
    newest entries once the thread passes ``per_page`` comments.
    """
    items: list = []
    sep = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = gh(f"{path}{sep}per_page={per_page}&page={page}") or []
        items.extend(batch)
        if len(batch) < per_page:
            return items
        page += 1


RUN_Q = """query($e:String!,$p:String!,$r:String!){
  project(name:$p,entityName:$e){
    run(name:$r){ name displayName state createdAt heartbeatAt config summaryMetrics }
  }}"""

HIST_Q = """query($e:String!,$p:String!,$r:String!,$specs:[JSONString!]!){
  project(name:$p,entityName:$e){ run(name:$r){ sampledHistory(specs:$specs) } }}"""

RUNS_Q = """query($e:String!,$p:String!,$n:Int!){
  project(name:$p,entityName:$e){
    runs(first:$n,order:"-createdAt"){edges{node{
      displayName state createdAt heartbeatAt summaryMetrics }}}}}"""

LINEAGE_Q = """query($e:String!,$p:String!,$f:JSONString,$n:Int!){
  project(name:$p,entityName:$e){
    runs(first:$n,order:"createdAt",filters:$f){edges{node{
      displayName state createdAt heartbeatAt summaryMetrics }}}}}"""


def resolve_lineage() -> tuple[str | None, list[dict]]:
    """Find the live hero run and every segment of its training lineage.

    The production run is relaunched under a new id after each intervention
    (hero-20260819 -> hero-12d8b6f0-dee637 -> ... -> hero-ragged_a2a-nccl2307-ep-step81k),
    and every segment reports ``run_progress`` against the same step budget.
    ``global_step / run_progress`` recovers that budget, which separates the
    campaign from gates, one-rack diagnostics and forecast runs. Returns the
    current segment's id and all segments oldest first, so the first
    segment's ``created_at`` is when the campaign started.
    """
    filt = json.dumps(
        {
            "$and": [
                {"displayName": {"$regex": "^hero"}},
                {"summary_metrics.run_progress": {"$exists": True}},
            ]
        }
    )
    edges = gql(LINEAGE_Q, {"e": ENTITY, "p": PROJECT, "f": filt, "n": 200})["project"][
        "runs"
    ]["edges"]
    segments = []
    for edge in edges:
        node = edge["node"]
        summary = json.loads(node["summaryMetrics"] or "{}")
        step, progress = summary.get("global_step"), summary.get("run_progress")
        if not isinstance(step, (int, float)) or not progress:
            continue
        segments.append(
            {
                "name": node["displayName"],
                "state": node["state"],
                "created_at": node["createdAt"],
                "heartbeat_at": node["heartbeatAt"],
                "step": step,
                "budget": round(step / progress),
                "runtime_s": summary.get("_runtime"),
            }
        )
    if not segments:
        return None, []
    # A finished campaign must not shadow a newer one, so pick the budget from
    # runs that were alive in the last week when there are any.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    pool = [s for s in segments if s["heartbeat_at"] >= cutoff[:19]] or segments
    budget = max(s["budget"] for s in pool)
    lineage = [s for s in segments if abs(s["budget"] - budget) <= budget * 0.001]
    current = max(
        lineage, key=lambda s: (s["state"] == "running", s["heartbeat_at"], s["step"])
    )
    return current["name"], lineage


def unwrap(cfg: dict) -> dict:
    """W&B nests every config value under {'value': ...}; flatten to dotted keys."""
    flat: dict = {}

    def walk(node, prefix=""):
        if not isinstance(node, dict):
            flat[prefix] = node
            return
        if "value" in node and len(node) <= 2:
            walk(node["value"], prefix)
            return
        for key, val in node.items():
            walk(val, f"{prefix}.{key}" if prefix else key)

    walk(cfg)
    return flat


def collect_hero(run_id: str) -> dict:
    print(f"  wandb: run {run_id}", file=sys.stderr)
    node = gql(RUN_Q, {"e": ENTITY, "p": PROJECT, "r": run_id})["project"]["run"]
    summary = json.loads(node["summaryMetrics"] or "{}")
    config = unwrap(json.loads(node["config"] or "{}"))

    print("  wandb: loss history", file=sys.stderr)
    loss_spec = json.dumps(
        {"keys": ["global_step", "train/loss", "_timestamp"], "samples": 600}
    )
    loss = gql(
        HIST_Q, {"e": ENTITY, "p": PROJECT, "r": run_id, "specs": [loss_spec]}
    )["project"]["run"]["sampledHistory"]

    print("  wandb: throughput + moe history", file=sys.stderr)
    health_spec = json.dumps(
        {
            "keys": [
                "global_step",
                "throughput/mfu",
                "throughput/tokens_per_second",
                "moe/drop_fraction",
                "optim/learning_rate",
                "params/norm/total",
            ],
            "samples": 400,
        }
    )
    health = gql(
        HIST_Q, {"e": ENTITY, "p": PROJECT, "r": run_id, "specs": [health_spec]}
    )["project"]["run"]["sampledHistory"]

    print("  wandb: eval history", file=sys.stderr)
    # sampledHistory only returns rows where EVERY requested key is present, so
    # asking for a key the run never logged yields zero points. Older runs logged
    # routed and dropless evals; the current run logs dropless only -- probe the
    # summary and request just the keys this run actually has.
    wanted = [
        "eval/bpb",
        "eval/macro_bpb",
        "eval/paloma/bpb",
        "eval/uncheatable_eval/bpb",
        "eval_dropless/bpb",
        "eval_dropless/macro_bpb",
        "eval_dropless/paloma/bpb",
        "eval_dropless/uncheatable_eval/bpb",
    ]
    eval_spec = json.dumps(
        {"keys": ["global_step"] + [k for k in wanted if k in summary], "samples": 200}
    )
    evals = gql(
        HIST_Q, {"e": ENTITY, "p": PROJECT, "r": run_id, "specs": [eval_spec]}
    )["project"]["run"]["sampledHistory"]

    return {
        "id": run_id,
        "display_name": node["displayName"],
        "state": node["state"],
        "created_at": node["createdAt"],
        "heartbeat_at": node["heartbeatAt"],
        "summary": summary,
        "config": config,
        "loss_history": loss[0] if loss else [],
        "health_history": health[0] if health else [],
        "eval_history": evals[0] if evals else [],
    }


def collect_recent_runs(limit: int = 60) -> list[dict]:
    print(f"  wandb: {limit} most recent runs", file=sys.stderr)
    edges = gql(RUNS_Q, {"e": ENTITY, "p": PROJECT, "n": limit})["project"]["runs"][
        "edges"
    ]
    rows = []
    for edge in edges:
        node = edge["node"]
        summary = json.loads(node["summaryMetrics"] or "{}")
        rows.append(
            {
                "name": node["displayName"],
                "state": node["state"],
                "created_at": node["createdAt"],
                "heartbeat_at": node["heartbeatAt"],
                "step": summary.get("global_step") or summary.get("_step"),
                "loss": summary.get("train/loss"),
                "mfu": summary.get("throughput/mfu"),
                "runtime_s": summary.get("_runtime"),
            }
        )
    return rows


def collect_github() -> dict:
    out: dict = {}

    print("  github: incident log", file=sys.stderr)
    comments = gh_all(f"repos/{REPO}/issues/{STATUS_ISSUE}/comments")
    out["status_log"] = [
        {
            "author": c["user"]["login"],
            "created_at": c["created_at"],
            "body": c["body"],
            "url": c["html_url"],
        }
        for c in comments
    ]

    print("  github: burndown", file=sys.stderr)
    issue = gh(f"repos/{REPO}/issues/{BURNDOWN_ISSUE}")
    gates = []
    if issue:
        for line in (issue.get("body") or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(("- [x]", "- [X]", "- [ ]")):
                gates.append(
                    {
                        "done": stripped[3] in "xX",
                        "text": stripped[5:].strip(),
                    }
                )
    out["burndown"] = gates

    print("  github: recent commits", file=sys.stderr)
    commits = gh(f"repos/{REPO}/commits?per_page=40") or []
    out["commits"] = [
        {
            "sha": c["sha"][:9],
            "message": (c["commit"]["message"].splitlines() or [""])[0],
            "author": (c.get("author") or {}).get("login")
            or c["commit"]["author"]["name"],
            "date": c["commit"]["author"]["date"],
            "url": c["html_url"],
        }
        for c in commits
    ]

    print("  github: open experiment issues", file=sys.stderr)
    issues = (
        gh(
            f"repos/{REPO}/issues?state=open&labels=experiment"
            "&sort=updated&direction=desc&per_page=30"
        )
        or []
    )
    out["experiments"] = [
        {
            "number": i["number"],
            "title": i["title"],
            "updated_at": i["updated_at"],
            "comments": i["comments"],
            "assigned": bool(i["assignees"]),
            "labels": [lbl["name"] for lbl in i["labels"]],
            "url": i["html_url"],
        }
        for i in issues
        if "pull_request" not in i
    ]

    print("  github: help-wanted issues", file=sys.stderr)
    helpw = (
        gh(f"repos/{REPO}/issues?state=open&labels=help-wanted&per_page=30") or []
    )
    out["help_wanted"] = [
        {
            "number": i["number"],
            "title": i["title"],
            "updated_at": i["updated_at"],
            "comments": i["comments"],
            "url": i["html_url"],
        }
        for i in helpw
        if "pull_request" not in i
    ]

    out["issue_urls"] = {
        "status": f"https://github.com/{REPO}/issues/{STATUS_ISSUE}",
        "spec": f"https://github.com/{REPO}/issues/{SPEC_ISSUE}",
        "burndown": f"https://github.com/{REPO}/issues/{BURNDOWN_ISSUE}",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="default: auto-detect the live hero run")
    parser.add_argument("--out", default="data.json")
    args = parser.parse_args()

    print("collecting...", file=sys.stderr)
    current, lineage = resolve_lineage()
    run_id = args.run or current
    if not run_id:
        sys.exit("could not resolve a hero run; pass --run explicitly")
    print(
        f"  hero run: {run_id} ({len(lineage)} runs in lineage since "
        f"{lineage[0]['created_at'] if lineage else '?'})",
        file=sys.stderr,
    )
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lineage": lineage,
        "hero": collect_hero(run_id),
        "recent_runs": collect_recent_runs(),
        "github": collect_github(),
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    size_kb = len(json.dumps(payload)) / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
