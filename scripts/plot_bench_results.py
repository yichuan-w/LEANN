#!/usr/bin/env python3
"""
Plot latency bars from the benchmark CSV produced by
examples/bench_hnsw_rng_recompute.py.

Usage:
  uv run scripts/plot_bench_results.py --csv bench_results.csv \
      --out bench_latency_from_csv.png

The script selects the latest run_id in the CSV and plots four bars for
the default scenarios:
  - baseline
  - no_cache_baseline
  - disable_forward_rng
  - disable_forward_and_reverse_rng

If multiple rows exist per scenario for that run_id, the script averages
their latency_ms_per_passage values.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_SCENARIOS = [
    "no_cache_baseline",
    "baseline",
    "disable_forward_rng",
    "disable_forward_and_reverse_rng",
]

SCENARIO_LABELS = {
    "baseline": "Cache",
    "no_cache_baseline": "Baseline",
    "disable_forward_rng": "No Forward RNG",
    "disable_forward_and_reverse_rng": "No RNG",
}


def load_latest_run(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit("CSV is empty: no rows to plot")
    # Choose latest run_id lexicographically (YYYYMMDD-HHMMSS)
    run_ids = [r.get("run_id", "") for r in rows]
    latest = max(run_ids)
    latest_rows = [r for r in rows if r.get("run_id", "") == latest]
    if not latest_rows:
        # Fallback: take last 4 rows
        latest_rows = rows[-4:]
        latest = latest_rows[-1].get("run_id", "unknown")
    return latest, latest_rows


def aggregate_latency(rows):
    acc = defaultdict(list)
    for r in rows:
        sc = r.get("scenario", "")
        try:
            val = float(r.get("latency_ms_per_passage", "nan"))
        except ValueError:
            continue
        acc[sc].append(val)
    avg = {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}
    return avg


def _auto_cap(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values, reverse=True)
    if len(sorted_vals) < 2:
        return None
    max_v, second = sorted_vals[0], sorted_vals[1]
    if second <= 0:
        return None
    # If the tallest bar dwarfs the second by 2.5x+, cap near the second
    if max_v >= 2.5 * second:
        return second * 1.1
    return None


def _add_break_marker(ax, y, rel_x0=0.02, rel_x1=0.98, size=0.02):
    # Draw small diagonal ticks near left/right to signal cap
    x0, x1 = rel_x0, rel_x1
    ax.plot([x0 - size, x0 + size], [y + size, y - size], transform=ax.transAxes, color="k", lw=1)
    ax.plot([x1 - size, x1 + size], [y + size, y - size], transform=ax.transAxes, color="k", lw=1)


def _fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True, help="Path to results CSV")
    ap.add_argument(
        "--out", type=Path, default=Path("bench_latency_from_csv.png"), help="Output image path"
    )
    ap.add_argument(
        "--cap-y",
        type=float,
        default=None,
        help="Cap Y-axis at this ms value; bars above are hatched and annotated.",
    )
    ap.add_argument(
        "--no-auto-cap",
        action="store_true",
        help="Disable auto-cap heuristic when --cap-y is not provided.",
    )
    ap.add_argument(
        "--broken-y",
        action="store_true",
        help="Use a broken Y-axis (two stacked axes with a gap). Overrides --cap-y unless both provided.",
    )
    ap.add_argument(
        "--lower-cap-y",
        type=float,
        default=None,
        help="Lower axes upper bound for broken Y (ms). Default = 1.1×second-highest.",
    )
    ap.add_argument(
        "--upper-start-y",
        type=float,
        default=None,
        help="Upper axes lower bound for broken Y (ms). Default = 1.2×second-highest.",
    )
    args = ap.parse_args()

    latest_run, latest_rows = load_latest_run(args.csv)
    avg = aggregate_latency(latest_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib not available: {e}")

    scenarios = DEFAULT_SCENARIOS
    values = [avg.get(name, 0.0) for name in scenarios]
    labels = [SCENARIO_LABELS.get(name, name) for name in scenarios]
    colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]

    # Broken-Y mode
    if args.broken_y:
        import matplotlib.pyplot as plt

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10, 6),
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
        )

        # Determine default breaks from second-highest
        s = sorted(values, reverse=True)
        second = s[1] if len(s) >= 2 else (s[0] if s else 0.0)
        lower_cap = args.lower_cap_y if args.lower_cap_y is not None else second * 1.1
        upper_start = (
            args.upper_start_y
            if args.upper_start_y is not None
            else max(second * 1.2, lower_cap * 1.02)
        )
        ymax = max(values) * 1.10 if values else 1.0

        x = list(range(len(labels)))
        ax_bottom.bar(x, values, color=colors[: len(labels)], width=0.8)
        ax_top.bar(x, values, color=colors[: len(labels)], width=0.8)

        # Limits
        ax_bottom.set_ylim(0, lower_cap)
        ax_top.set_ylim(upper_start, ymax)

        # Annotate values
        for i, v in enumerate(values):
            if v <= lower_cap:
                ax_bottom.text(
                    i, v + lower_cap * 0.02, _fmt_ms(v), ha="center", va="bottom", fontsize=9
                )
            else:
                ax_top.text(i, v, _fmt_ms(v), ha="center", va="bottom", fontsize=9)

        # Hide spines between axes and draw diagonal break marks
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
        ax_bottom.xaxis.tick_bottom()

        # Diagonal lines at the break (Matplotlib gallery style)
        d = 0.015
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels(labels, rotation=0, fontsize=10)
        ax = ax_bottom  # for labeling below
    else:
        cap = args.cap_y
        if cap is None and not args.no_auto_cap:
            cap = _auto_cap(values)

        plt.figure(figsize=(7.2, 4.2))
        ax = plt.gca()

        if cap is not None:
            show_vals = [min(v, cap) for v in values]
            bars = []
            for i, (lab, val, show) in enumerate(zip(labels, values, show_vals)):
                bar = ax.bar(i, show, color=colors[i], width=0.8)
                bars.append(bar[0])
                # Hatch and annotate when capped
                if val > cap:
                    bars[-1].set_hatch("//")
                    ax.text(i, cap * 1.02, f"{_fmt_ms(val)}", ha="center", va="bottom", fontsize=9)
                else:
                    ax.text(
                        i,
                        show + max(1.0, 0.01 * (cap or show)),
                        f"{_fmt_ms(val)}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
            ax.set_ylim(0, cap * 1.10)
            _add_break_marker(ax, y=0.98)
            ax.legend([bars[1]], ["capped"], fontsize=8, frameon=False, loc="upper right") if any(
                v > cap for v in values
            ) else None
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
        else:
            ax.bar(labels, values, color=colors[: len(labels)])
            for idx, val in enumerate(values):
                ax.text(idx, val + 1.0, f"{_fmt_ms(val)}", ha="center", va="bottom")
    # Try to extract some context for title
    max_initial = latest_rows[0].get("max_initial", "?")
    max_updates = latest_rows[0].get("max_updates", "?")

    if args.broken_y:
        fig.text(
            0.02, 0.5, "Latency (ms per passage)", va="center", rotation="vertical", fontsize=11
        )
        fig.suptitle(
            f"Latency bars (run {latest_run}) | initial={max_initial}, updates={max_updates}",
            fontsize=12,
            y=0.98,
        )
        plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    else:
        plt.ylabel("Latency (ms per passage)")
        plt.title(f"Latency bars (run {latest_run}) | initial={max_initial}, updates={max_updates}")
        plt.tight_layout()

    plt.savefig(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
