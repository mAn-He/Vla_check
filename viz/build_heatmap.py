"""Build the two success-rate heatmaps from results/*.jsonl.

(a) summary:  rows = models, cols = task suites (+ Average), % values annotated
(b) per-task: rows = models, cols = every task, columns sorted ascending by the
    across-model mean success rate, with separators at suite boundaries.
    This is the main figure: sorted columns expose which tasks collapse for
    every model and where a single model is unusually strong.

Both use a sequential colormap on a fixed 0-100 scale and are written as
PNG + SVG into viz/figures/.

`--demo` writes dummy JSONL files (three fake models, all four suites) into a
temp directory and renders from those — the full pipeline runs without a GPU:
    python viz/build_heatmap.py --demo
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CMAP = "viridis"  # sequential (NOT diverging), per project spec


def load_records(results_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[warn] skipping unparseable line in {path}", file=sys.stderr)
    if not rows:
        sys.exit(f"no episode records found under {results_dir}")
    df = pd.DataFrame(rows)
    df["success"] = df["success"].astype(bool)
    return df


def summary_heatmap(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    suites = [s for s in SUITE_ORDER if s in set(df["task_suite"])] or sorted(set(df["task_suite"]))
    rate = (
        df.groupby(["model", "task_suite"])["success"].mean().mul(100).unstack("task_suite").reindex(columns=suites)
    )
    rate["Average"] = rate.mean(axis=1)
    fig, ax = plt.subplots(figsize=(2.2 + 1.6 * len(rate.columns), 1.2 + 0.7 * len(rate.index)))
    sns.heatmap(rate, annot=True, fmt=".1f", vmin=0, vmax=100, cmap=CMAP,
                cbar_kws={"label": "success rate (%)"}, linewidths=0.5, ax=ax)
    ax.set_title("LIBERO success rate by task suite (%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, out_dir, "heatmap_summary")


def per_task_heatmap(df: pd.DataFrame, out_dir: pathlib.Path, sort: str = "global") -> None:
    """sort="global": every task ascending by across-model mean (the spec's main
    view); suites can no longer stay contiguous, so suite membership is shown as
    a colored band above the heatmap instead of separator lines.
    sort="suite":  suite blocks in canonical order with separator lines at the
    boundaries, ascending within each block."""
    suites = [s for s in SUITE_ORDER if s in set(df["task_suite"])] or sorted(set(df["task_suite"]))
    df = df[df["task_suite"].isin(suites)].copy()
    df["col"] = df["task_suite"] + "/" + df["task_id"].astype(str)
    rate = df.groupby(["model", "col"])["success"].mean().mul(100).unstack("col")

    boundaries = []
    if sort == "suite":
        ordered_cols = []
        for suite in suites:
            cols = [c for c in rate.columns if c.startswith(suite + "/")]
            cols.sort(key=lambda c: (rate[c].mean(), c))
            ordered_cols.extend(cols)
            boundaries.append(len(ordered_cols))
    else:
        ordered_cols = sorted(rate.columns, key=lambda c: (rate[c].mean(), c))
    rate = rate[ordered_cols]

    labels = []
    task_names = df.drop_duplicates("col").set_index("col")["task_name"]
    for c in ordered_cols:
        name = str(task_names.get(c, c))
        labels.append(name if len(name) <= 40 else name[:37] + "...")

    fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(ordered_cols)), 2.0 + 0.7 * len(rate.index)))
    sns.heatmap(rate, vmin=0, vmax=100, cmap=CMAP, cbar_kws={"label": "success rate (%)"}, ax=ax)
    if sort == "suite":
        for b in boundaries[:-1]:
            ax.axvline(b, color="white", linewidth=2.5)
    else:
        # suite band above the heatmap (one flat color per suite)
        band_colors = dict(zip(suites, sns.color_palette("pastel", n_colors=len(suites))))
        for i, c in enumerate(ordered_cols):
            suite = c.rsplit("/", 1)[0]
            ax.add_patch(plt.Rectangle((i, -0.28), 1, 0.24, clip_on=False,
                                       facecolor=band_colors[suite], edgecolor="none"))
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=band_colors[s]) for s in suites]
        ax.legend(handles, suites, loc="lower left", bbox_to_anchor=(0, 1.05), ncol=len(suites),
                  frameon=False, fontsize=8)
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_title(f"LIBERO success rate per task (%), ascending by across-model mean (sort={sort})", pad=30)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, out_dir, "heatmap_per_task")


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"[viz] wrote {path}")
    plt.close(fig)


def write_demo_results(demo_dir: pathlib.Path) -> None:
    """Dummy JSONL in the exact episode schema, for pipeline validation."""
    rng = np.random.default_rng(0)
    demo_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10}
    for model, base in [("demo_model_a", 0.9), ("demo_model_b", 0.7), ("demo_model_c", 0.4)]:
        for suite, n in n_tasks.items():
            with open(demo_dir / f"{model}__{suite}.jsonl", "w") as f:
                for task_id in range(n):
                    p = np.clip(base + rng.uniform(-0.35, 0.1), 0, 1)
                    for ep in range(10):
                        success = bool(rng.random() < p)
                        rec = {
                            "model": model,
                            "model_source": "demo",
                            "task_suite": suite,
                            "task_id": task_id,
                            "task_name": f"{suite} demo task {task_id}",
                            "episode_idx": ep,
                            "seed": 42,
                            "success": success,
                            "num_steps": int(rng.integers(30, 200)),
                            "max_steps": 220,
                            "wall_time_sec": float(np.round(rng.uniform(5, 40), 2)),
                            "error": None,
                        }
                        f.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "viz" / "figures"))
    parser.add_argument("--demo", action="store_true", help="render from generated dummy results (no GPU needed)")
    parser.add_argument("--sort", choices=("global", "suite"), default="global",
                        help="per-task column order: global ascending (suite shown as color band) "
                             "or per-suite blocks with boundary lines")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    if args.demo:
        results_dir = REPO_ROOT / "results" / "demo"
        write_demo_results(results_dir)

    df = load_records(results_dir)
    out_dir = pathlib.Path(args.out_dir)
    summary_heatmap(df, out_dir)
    per_task_heatmap(df, out_dir, sort=args.sort)


if __name__ == "__main__":
    main()
