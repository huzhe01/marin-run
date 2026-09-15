# marin-run

A self-refreshing, unofficial dashboard for [Marin](https://marin.community/)'s 535B-A23B MoE pretraining run, built entirely from public data.

**Live site: https://huzhe01.github.io/marin-run/**

- **实时看板** — progress against the 390,251-step budget, loss / MFU / routing-drop / eval curves, per-slice Paloma and Uncheatable bpb, per-layer router overflow, the live 40 × 5 data-mixture weights, the run's relaunch lineage, the incident log from [marin#8506](https://github.com/marin-community/marin/issues/8506), launch gates, recent commits and open contribution issues.
- **全流程拆解** — a walkthrough of the whole pipeline (data, scaling ladder, pretraining, post-training, evals, infrastructure, operations), dated 2026-08-30.
- **更新报告** — hand-written summaries of what changed, one Markdown file per date in [`reports/`](reports/).

This project is not affiliated with Marin or Open Athena. Everything it shows comes from the public [`marin-community/marin_moe`](https://wandb.ai/marin-community/marin_moe) W&B project and the public [`marin-community/marin`](https://github.com/marin-community/marin) GitHub repository; no credentials are used to read either.

## How it refreshes

[`.github/workflows/refresh.yml`](.github/workflows/refresh.yml) runs every 6 hours (and on demand):

1. `collect.py` reads the live run from W&B and the incident log, launch gates, commits and issues from GitHub, and writes `data.json`.
2. `build_site.py` renders the dashboard, the explainer and the reports into `site/`, and appends one row to [`data/history.csv`](data/history.csv).
3. The site is deployed to GitHub Pages, and the history row is committed.

The history commit also keeps the schedule alive: GitHub disables scheduled workflows in public repositories after 60 days without activity. If a run fails (for example W&B is unreachable), the previous deployment stays up and GitHub emails the repository owner.

To refresh immediately:

```bash
gh workflow run refresh --repo huzhe01/marin-run
```

## Finding the run across relaunches

The production run is relaunched under a new W&B run id after each intervention (`hero-20260819` → `hero-12d8b6f0-dee637` → … → `hero-ragged_a2a-nccl2307-ep-step81k`). Every segment reports `run_progress` against the same step budget, so `collect.py` recovers each run's budget as `global_step / run_progress` and treats runs sharing the campaign's budget as one lineage. Wall-clock progress and the ETA are measured from the first segment, not the current one.

## Run locally

```bash
python3 collect.py --out data.json
uv run --with markdown python build_site.py --data data.json --out site
open site/index.html
```

`collect.py` uses only the standard library and, when available, the `gh` CLI; `build_site.py` additionally needs `markdown`.

## Adding a report

Add `reports/YYYY-MM-DD.md`. The first `# ` heading becomes the page title; the rest is rendered with Markdown tables and fenced code. The next build lists it on the reports page.
