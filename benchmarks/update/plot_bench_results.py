#!/usr/bin/env python3
"""
Plot latency bars from the benchmark CSV produced by
benchmarks/update/bench_hnsw_rng_recompute.py.

If you also provide an offline_vs_update.csv via --csv-right
(from benchmarks/update/bench_update_vs_offline_search.py), this script will
output a side-by-side figure:
- Left: ms/passage bars (four RNG scenarios).
- Right: seconds bars (Scenario A seq add+search vs Scenario B offline+search).

Usage:
  uv run python benchmarks/update/plot_bench_results.py \
      --csv benchmarks/update/bench_results.csv \
      --out benchmarks/update/bench_latency_from_csv.png

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
    "baseline": "+ Cache",
    "no_cache_baseline": "Naive \n Recompute",
    "disable_forward_rng": "+ w/o \n Fwd RNG",
    "disable_forward_and_reverse_rng": "+ w/o \n Bwd RNG",
}

# Paper-style colors and hatches for scenarios
SCENARIO_STYLES = {
    "no_cache_baseline": {"edgecolor": "dimgrey", "hatch": "/////"},
    "baseline": {"edgecolor": "#63B8B6", "hatch": "xxxxx"},
    "disable_forward_rng": {"edgecolor": "green", "hatch": "....."},
    "disable_forward_and_reverse_rng": {"edgecolor": "tomato", "hatch": "\\\\\\\\\\"},
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
    # Set LaTeX style for paper figures (matching paper_fig.py)
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["hatch.linewidth"] = 1.5
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["text.usetex"] = True

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/update/bench_results.csv"),
        help="Path to results CSV (defaults to bench_results.csv)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("add_ablation.pdf"),
        help="Output image path",
    )
    ap.add_argument(
        "--csv-right",
        type=Path,
        default=Path("benchmarks/update/offline_vs_update.csv"),
        help="Optional: offline_vs_update.csv to render right subplot (A vs B)",
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
        default=True,
        help="Use a broken Y-axis (two stacked axes with a gap). Overrides --cap-y unless both provided.",
    )
    ap.add_argument(
        "--lower-cap-y",
        type=float,
        default=None,
        help="Lower axes upper bound for broken Y (ms). Default = 1.1x second-highest.",
    )
    ap.add_argument(
        "--upper-start-y",
        type=float,
        default=None,
        help="Upper axes lower bound for broken Y (ms). Default = 1.2x second-highest.",
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

    # If right CSV is provided, build side-by-side figure
    if args.csv_right is not None:
        try:
            right_rows_all = []
            with args.csv_right.open("r", encoding="utf-8") as f:
                rreader = csv.DictReader(f)
                right_rows_all = list(rreader)
            if right_rows_all:
                r_latest = max(r.get("run_id", "") for r in right_rows_all)
                right_rows = [r for r in right_rows_all if r.get("run_id", "") == r_latest]
            else:
                r_latest = None
                right_rows = []
        except Exception:
            r_latest = None
            right_rows = []

        a_total = 0.0
        b_makespan = 0.0
        for r in right_rows:
            sc = (r.get("scenario", "") or "").strip().upper()
            if sc == "A":
                try:
                    a_total = float(r.get("total_time_s", 0.0))
                except Exception:
                    pass
            elif sc == "B":
                try:
                    b_makespan = float(r.get("makespan_s", 0.0))
                except Exception:
                    pass

        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # Left subplot (reuse current style, with optional cap)
        cap = args.cap_y
        if cap is None and not args.no_auto_cap:
            cap = _auto_cap(values)
        x = list(range(len(labels)))

        if args.broken_y:
            # Use broken axis for left subplot
            # Auto-adjust width ratios: left has 4 bars, right has 2 bars
            fig = plt.figure(figsize=(4.8, 1.8))  # Scaled down to 80%
            gs = gridspec.GridSpec(
                2, 2, height_ratios=[1, 3], width_ratios=[1.5, 1], hspace=0.08, wspace=0.35
            )
            ax_left_top = fig.add_subplot(gs[0, 0])
            ax_left_bottom = fig.add_subplot(gs[1, 0], sharex=ax_left_top)
            ax_right = fig.add_subplot(gs[:, 1])

            # Determine break points
            s = sorted(values, reverse=True)
            second = s[1] if len(s) >= 2 else (s[0] if s else 0.0)
            lower_cap = (
                args.lower_cap_y if args.lower_cap_y is not None else second * 1.4
            )  # Increased to show more range
            upper_start = (
                args.upper_start_y
                if args.upper_start_y is not None
                else max(second * 1.5, lower_cap * 1.02)
            )
            ymax = (
                max(values) * 1.90 if values else 1.0
            )  # Increase headroom to 1.90 for text label and tick range

            # Draw bars on both axes
            ax_left_bottom.bar(x, values, color=colors[: len(labels)], width=0.8)
            ax_left_top.bar(x, values, color=colors[: len(labels)], width=0.8)

            # Set limits
            ax_left_bottom.set_ylim(0, lower_cap)
            ax_left_top.set_ylim(upper_start, ymax)

            # Annotate values (convert ms to s)
            values_s = [v / 1000.0 for v in values]
            lower_cap_s = lower_cap / 1000.0
            upper_start_s = upper_start / 1000.0
            ymax_s = ymax / 1000.0

            ax_left_bottom.set_ylim(0, lower_cap_s)
            ax_left_top.set_ylim(upper_start_s, ymax_s)

            # Redraw bars with s values (paper style: white fill + colored edge + hatch)
            ax_left_bottom.clear()
            ax_left_top.clear()
            bar_width = 0.50  # Reduced for wider spacing between bars
            for i, (scenario_name, v) in enumerate(zip(scenarios, values_s)):
                style = SCENARIO_STYLES.get(scenario_name, {"edgecolor": "black", "hatch": ""})
                # Draw in bottom axis for all bars
                ax_left_bottom.bar(
                    i,
                    v,
                    width=bar_width,
                    color="white",
                    edgecolor=style["edgecolor"],
                    hatch=style["hatch"],
                    linewidth=1.2,
                )
                # Only draw in top axis if the bar is tall enough to reach the upper range
                if v > upper_start_s:
                    ax_left_top.bar(
                        i,
                        v,
                        width=bar_width,
                        color="white",
                        edgecolor=style["edgecolor"],
                        hatch=style["hatch"],
                        linewidth=1.2,
                    )
            ax_left_bottom.set_ylim(0, lower_cap_s)
            ax_left_top.set_ylim(upper_start_s, ymax_s)

            for i, v in enumerate(values_s):
                if v <= lower_cap_s:
                    ax_left_bottom.text(
                        i,
                        v + lower_cap_s * 0.02,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )
                else:
                    ax_left_top.text(
                        i,
                        v + (ymax_s - upper_start_s) * 0.02,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

            # Hide spines between axes
            ax_left_top.spines["bottom"].set_visible(False)
            ax_left_bottom.spines["top"].set_visible(False)
            ax_left_top.tick_params(
                labeltop=False, labelbottom=False, bottom=False
            )  # Hide tick marks
            ax_left_bottom.xaxis.tick_bottom()
            ax_left_bottom.tick_params(top=False)  # Hide top tick marks

            # Draw break marks (matching paper_fig.py style)
            d = 0.015
            kwargs = {
                "transform": ax_left_top.transAxes,
                "color": "k",
                "clip_on": False,
                "linewidth": 0.8,
                "zorder": 10,
            }
            ax_left_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_left_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update({"transform": ax_left_bottom.transAxes})
            ax_left_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_left_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            ax_left_bottom.set_xticks(x)
            ax_left_bottom.set_xticklabels(labels, rotation=0, fontsize=7)
            # Don't set ylabel here - will use fig.text for alignment
            ax_left_bottom.tick_params(axis="y", labelsize=10)
            ax_left_top.tick_params(axis="y", labelsize=10)
            # Add subtle grid for better readability
            ax_left_bottom.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
            ax_left_top.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
            ax_left_top.set_title("Single Add Operation", fontsize=11, pad=10, fontweight="bold")

            # Set x-axis limits to match bar width with right subplot
            ax_left_bottom.set_xlim(-0.6, 3.6)
            ax_left_top.set_xlim(-0.6, 3.6)

            ax_left = ax_left_bottom  # for compatibility
        else:
            # Regular side-by-side layout
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8.4, 3.15))

            if cap is not None:
                show_vals = [min(v, cap) for v in values]
                bars = ax_left.bar(x, show_vals, color=colors[: len(labels)], width=0.8)
                for i, (val, show) in enumerate(zip(values, show_vals)):
                    if val > cap:
                        bars[i].set_hatch("//")
                        ax_left.text(
                            i, cap * 1.02, _fmt_ms(val), ha="center", va="bottom", fontsize=9
                        )
                    else:
                        ax_left.text(
                            i,
                            show + max(1.0, 0.01 * (cap or show)),
                            _fmt_ms(val),
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )
                ax_left.set_ylim(0, cap * 1.10)
                _add_break_marker(ax_left, y=0.98)
                ax_left.set_xticks(x)
                ax_left.set_xticklabels(labels, rotation=0, fontsize=10)
            else:
                ax_left.bar(x, values, color=colors[: len(labels)], width=0.8)
                for i, v in enumerate(values):
                    ax_left.text(i, v + 1.0, _fmt_ms(v), ha="center", va="bottom", fontsize=9)
                ax_left.set_xticks(x)
                ax_left.set_xticklabels(labels, rotation=0, fontsize=10)
            ax_left.set_ylabel("Latency (ms per passage)")
            max_initial = latest_rows[0].get("max_initial", "?")
            max_updates = latest_rows[0].get("max_updates", "?")
            ax_left.set_title(
                f"HNSW RNG (run {latest_run}) | init={max_initial}, upd={max_updates}"
            )

        # Right subplot (A vs B, seconds) - paper style
        r_labels = ["Sequential", "Delayed \n Add+Search"]
        r_values = [a_total or 0.0, b_makespan or 0.0]
        r_styles = [
            {"edgecolor": "#59a14f", "hatch": "xxxxx"},
            {"edgecolor": "#edc948", "hatch": "/////"},
        ]
        # 2 bars, centered with proper spacing
        xr = [0, 1]
        bar_width = 0.50  # Reduced for wider spacing between bars
        for i, (v, style) in enumerate(zip(r_values, r_styles)):
            ax_right.bar(
                xr[i],
                v,
                width=bar_width,
                color="white",
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=1.2,
            )
        for i, v in enumerate(r_values):
            max_v = max(r_values) if r_values else 1.0
            offset = max(0.0002, 0.02 * max_v)
            ax_right.text(
                xr[i],
                v + offset,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax_right.set_xticks(xr)
        ax_right.set_xticklabels(r_labels, rotation=0, fontsize=7)
        # Don't set ylabel here - will use fig.text for alignment
        ax_right.tick_params(axis="y", labelsize=10)
        # Add subtle grid for better readability
        ax_right.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax_right.set_title("Batched Add Operation", fontsize=11, pad=10, fontweight="bold")

        # Set x-axis limits to match left subplot's bar width visually
        # Accounting for width_ratios=[1.5, 1]:
        # Left: 4 bars, xlim(-0.6, 3.6), range=4.2, physical_width=1.5*unit
        # bar_width_visual = 0.72 * (1.5*unit / 4.2)
        # Right: 2 bars, need same visual width
        # 0.72 * (1.0*unit / range_right) = 0.72 * (1.5*unit / 4.2)
        # range_right = 4.2 / 1.5 = 2.8
        # For bars at 0, 1: padding = (2.8 - 1) / 2 = 0.9
        ax_right.set_xlim(-0.9, 1.9)

        # Set y-axis limit with headroom for text labels
        if r_values:
            max_v = max(r_values)
            ax_right.set_ylim(0, max_v * 1.15)

        # Format y-axis to avoid scientific notation
        ax_right.ticklabel_format(style="plain", axis="y")

        plt.tight_layout()

        # Add aligned ylabels using fig.text (after tight_layout)
        # Get the vertical center of the entire figure
        fig_center_y = 0.5
        # Left ylabel - closer to left plot
        left_x = 0.05
        fig.text(
            left_x,
            fig_center_y,
            "Latency (s)",
            va="center",
            rotation="vertical",
            fontsize=11,
            fontweight="bold",
        )
        # Right ylabel - closer to right plot
        right_bbox = ax_right.get_position()
        right_x = right_bbox.x0 - 0.07
        fig.text(
            right_x,
            fig_center_y,
            "Latency (s)",
            va="center",
            rotation="vertical",
            fontsize=11,
            fontweight="bold",
        )

        plt.savefig(args.out, bbox_inches="tight", pad_inches=0.05)
        # Also save PDF for paper
        pdf_out = args.out.with_suffix(".pdf")
        plt.savefig(pdf_out, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {args.out}")
        print(f"Saved: {pdf_out}")
        return

    # Broken-Y mode
    if args.broken_y:
        import matplotlib.pyplot as plt

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7.5, 6.75),
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

        # Diagonal lines at the break (matching paper_fig.py style)
        d = 0.015
        kwargs = {
            "transform": ax_top.transAxes,
            "color": "k",
            "clip_on": False,
            "linewidth": 0.8,
            "zorder": 10,
        }
        ax_top.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        kwargs.update({"transform": ax_bottom.transAxes})
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels(labels, rotation=0, fontsize=10)
        ax = ax_bottom  # for labeling below
    else:
        cap = args.cap_y
        if cap is None and not args.no_auto_cap:
            cap = _auto_cap(values)

        plt.figure(figsize=(5.4, 3.15))
        ax = plt.gca()

        if cap is not None:
            show_vals = [min(v, cap) for v in values]
            bars = []
            for i, (_label, val, show) in enumerate(zip(labels, values, show_vals)):
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
            ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        else:
            ax.bar(labels, values, color=colors[: len(labels)])
            for idx, val in enumerate(values):
                ax.text(
                    idx,
                    val + 1.0,
                    f"{_fmt_ms(val)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    # Try to extract some context for title
    max_initial = latest_rows[0].get("max_initial", "?")
    max_updates = latest_rows[0].get("max_updates", "?")

    if args.broken_y:
        fig.text(
            0.02,
            0.5,
            "Latency (s)",
            va="center",
            rotation="vertical",
            fontsize=11,
            fontweight="bold",
        )
        fig.suptitle(
            "Add Operation Latency",
            fontsize=11,
            y=0.98,
            fontweight="bold",
        )
        plt.tight_layout(rect=(0.03, 0.04, 1, 0.96))
    else:
        plt.ylabel("Latency (s)", fontsize=11, fontweight="bold")
        plt.title("Add Operation Latency", fontsize=11, fontweight="bold")
        plt.tight_layout()

    plt.savefig(args.out, bbox_inches="tight", pad_inches=0.05)
    # Also save PDF for paper
    pdf_out = args.out.with_suffix(".pdf")
    plt.savefig(pdf_out, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {args.out}")
    print(f"Saved: {pdf_out}")


if __name__ == "__main__":
    main()
