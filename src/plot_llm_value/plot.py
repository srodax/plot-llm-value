"""Headless Matplotlib rendering with label placement in display coordinates."""

import math
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from matplotlib.transforms import Bbox

from .data import SelectionError, effort_order

BACKGROUND = "#FBFAF7"
COLORS = ["#6B4F9B", "#D97745", "#2C7A7B", "#4C78A8", "#59A14F", "#B279A2", "#8C6D31", "#B64857"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]


def _place_labels(ax, requests, point_positions, segments):
    """Greedy nearest free candidate, measured after log/linear transforms.

    Text is placed only where its padded rectangle clears points, previous text
    and series segments. Search all group points for model labels; effort labels
    have one anchor. Dense inputs grow the canvas rather than hiding labels.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied, leaders = [], []
    bounds = ax.bbox.padded(-5)
    points = np.asarray(point_positions)
    line_paths = [MplPath(np.asarray(s)) for s in segments]
    for label, anchors, color, model_label in requests:
        text = ax.text(
            0,
            0,
            label,
            fontsize=9 if model_label else 7.5,
            weight="bold" if model_label else "normal",
            color=color,
            ha="center",
            va="center",
            zorder=5,
        )
        extent = text.get_window_extent(renderer)
        width, height = extent.width + 6, extent.height + 5
        anchor_pixels = ax.transData.transform(anchors)
        # A fine display-space lattice gives a deterministic fallback even when
        # many labels share exactly the same anchor.
        xs = np.arange(bounds.x0 + width / 2, bounds.x1 - width / 2, 10)
        ys = np.arange(bounds.y0 + height / 2, bounds.y1 - height / 2, 10)
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        if not len(candidates):
            return False
        distances = ((candidates[:, None, :] - anchor_pixels[None, :, :]) ** 2).sum(axis=2)
        closest = distances.argmin(axis=1)
        order = np.argsort(distances.min(axis=1), kind="stable")
        found = None
        for index in order:
            x, y = candidates[index]
            box = Bbox.from_bounds(x - width / 2, y - height / 2, width, height)
            if any(box.overlaps(other) for other in occupied):
                continue
            if len(points) and np.any(
                (points[:, 0] >= box.x0 - 9)
                & (points[:, 0] <= box.x1 + 9)
                & (points[:, 1] >= box.y0 - 9)
                & (points[:, 1] <= box.y1 + 9)
            ):
                continue
            if any(path.intersects_bbox(box.padded(2)) for path in line_paths):
                continue
            found = (x, y, box, anchor_pixels[closest[index]])
            break
        if found is None:
            return False
        x, y, box, anchor = found
        text.set_position(ax.transData.inverted().transform((x, y)))
        occupied.append(box)
        # Leaders terminate at the label edge; they do not cover the text.
        target = np.array([np.clip(anchor[0], box.x0, box.x1), np.clip(anchor[1], box.y0, box.y1)])
        if np.linalg.norm(target - anchor) > 15:
            leaders.append((target, anchor, color))
    # Clip leaders around every label rectangle. A dense cloud can completely
    # surround an anchor with text; forcing a clear straight ray would reject
    # otherwise readable layouts. Small gaps keep leaders from obscuring text.
    for target, anchor, color in leaders:
        delta = anchor - target
        hidden = []
        for box in occupied:
            box = box.padded(2)
            enter, leave = 0.0, 1.0
            for dim, (low, high) in enumerate(((box.x0, box.x1), (box.y0, box.y1))):
                if abs(delta[dim]) < 1e-10:
                    if not low <= target[dim] <= high:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    t1, t2 = (low - target[dim]) / delta[dim], (high - target[dim]) / delta[dim]
                    enter, leave = max(enter, min(t1, t2)), min(leave, max(t1, t2))
            if enter < leave:
                hidden.append((enter, leave))
        cursor = 0.0
        for start, end in sorted(hidden) + [(1.0, 1.0)]:
            if start > cursor and np.linalg.norm(delta * (start - cursor)) > 4:
                ax.add_patch(
                    FancyArrowPatch(
                        ax.transData.inverted().transform(target + cursor * delta),
                        ax.transData.inverted().transform(target + start * delta),
                        arrowstyle="-",
                        color=color,
                        lw=0.6,
                        alpha=0.5,
                        zorder=1,
                        shrinkA=0,
                        shrinkB=0,
                    )
                )
            cursor = max(cursor, end)
    return True


def build_figure(comparison, *, linear_x=False):
    groups, excluded = defaultdict(list), []
    for row in comparison["rows"]:
        reasons = list(row["exclusion_reasons"])
        if not linear_x and row["cost_per_task_usd"] == 0:
            reasons.append("zero cost cannot appear on log x; use --linear-x")
        if reasons:
            excluded.append({"id": row["id"], "name": row["name"], "reasons": reasons})
        else:
            groups[(row["provider_id"], row["family"])].append(row)
    if not groups:
        raise SelectionError(
            "No comparable measurements to plot; inspect data or use --linear-x for zero cost."
        )
    count = sum(map(len, groups.values()))
    label_count = count + len(groups)
    scale = max(1, math.sqrt(label_count / 45))
    providers = sorted({r["provider"] for rows in groups.values() for r in rows})
    marker_map = {p: MARKERS[i % len(MARKERS)] for i, p in enumerate(providers)}
    for expansion in (1, 1.3, 1.7):
        width, height = 10 * scale * expansion, 7.2 * scale * expansion
        fig, ax = plt.subplots(figsize=(width, height), dpi=180)
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(BACKGROUND)
        legend_rows = math.ceil(len(groups) / 4) + (
            math.ceil(len(providers) / 5) if len(providers) > 1 else 0
        )
        bottom = max(0.22, (0.85 + legend_rows * 0.19) / height)
        fig.subplots_adjust(left=0.095, right=0.96, top=0.85, bottom=min(bottom, 0.48))
        ax.set_xscale("linear" if linear_x else "log")
        requests, handles, coordinates, data_segments = [], [], [], []
        efforts = []
        for i, ((provider_id, family), rows) in enumerate(sorted(groups.items())):
            rows = sorted(rows, key=lambda r: (effort_order(r["effort"]), r["id"]))
            color = (
                COLORS[i % len(COLORS)]
                if len(groups) <= len(COLORS)
                else plt.cm.turbo((i + 0.5) / len(groups))
            )
            marker = marker_map[rows[0]["provider"]]
            xy = [(r["cost_per_task_usd"], r["score"]) for r in rows]
            coordinates.extend(xy)
            for start, end in zip(rows, rows[1:]):
                a, b = (
                    (start["cost_per_task_usd"], start["score"]),
                    (end["cost_per_task_usd"], end["score"]),
                )
                gap = effort_order(end["effort"])[0] - effort_order(start["effort"])[0] > 1
                (line,) = ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=color,
                    lw=2,
                    alpha=0.75,
                    linestyle="--" if gap else "-",
                    zorder=2,
                )
                line.set_gid(f"series:{provider_id}:{family}")
                data_segments.append((a, b))
            points = ax.scatter(
                *zip(*xy),
                s=65,
                marker=marker,
                color=color,
                edgecolor=BACKGROUND,
                linewidth=1.2,
                zorder=3,
            )
            points.set_gid(f"points:{provider_id}:{family}")
            requests.append((family, xy, color, True))
            for row, point in zip(rows, xy):
                if row["effort"]:
                    efforts.append((row["effort"], [point], color, False))
            handles.append(Line2D([], [], color=color, marker=marker, lw=2, label=family))
        ax.margins(x=0.18, y=0.2)
        if linear_x:
            low, high = ax.get_xlim()
            ax.set_xlim(max(0, low), high)
        ax.grid(True, which="major", color="#DEDCD6", lw=0.8)
        ax.grid(True, which="minor", axis="x", color="#ECEAE4", lw=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not linear_x:
            low, high = ax.get_xlim()
            decades = math.log10(high / low)
            if decades < 0.3:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            else:
                ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 3, 5) if decades < 2 else (1,)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=8)
        metric = comparison["metric"]["label"]
        ax.set_ylabel("Artificial Analysis " + metric, fontsize=9.5, labelpad=9)
        ax.set_xlabel(
            "Cost per Intelligence Index task (USD" + (", log scale)" if not linear_x else ")"),
            fontsize=9.5,
            labelpad=9,
        )
        fig.text(0.095, 0.953, metric + " vs. cost per task", weight="bold", fontsize=16, ha="left")
        fig.text(
            0.095,
            0.914,
            "Reasoning settings connected within each release · Higher and further left is better",
            color="#555555",
            fontsize=8.5,
        )
        if len(providers) > 1:
            handles.extend(
                Line2D([], [], color="#555555", marker=marker_map[p], lw=0, label=p)
                for p in providers
            )
        source = comparison.get("source", {})
        fig.text(
            0.095,
            0.025,
            "Source: Artificial Analysis · https://artificialanalysis.ai/\n"
            f"Retrieved {source.get('retrieved_at', 'unknown')} · Intelligence Index v{source.get('intelligence_index_version', '?')}\n"
            f"{count} / {len(comparison['rows'])} rows plotted · Dashed: effort gap · Costs use Intelligence Index tasks",
            fontsize=6.5,
            color="#666666",
            linespacing=1.4,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        footer_top = fig.texts[-1].get_window_extent(renderer).y1
        max_label_width = max(
            renderer.get_text_width_height_descent(
                handle.get_label(), FontProperties(size=7.2), False
            )[0]
            for handle in handles
        )
        columns = max(1, min(4, int(fig.bbox.width * 0.865 / (max_label_width + 90))))
        legend = ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.095, (footer_top + 12) / fig.bbox.height),
            bbox_transform=fig.transFigure,
            ncol=columns,
            frameon=False,
            fontsize=7.2,
            columnspacing=1.1,
            borderaxespad=0,
        )
        fig.canvas.draw()
        legend_top = legend.get_window_extent(renderer).y1
        bottom = (legend_top + 110) / fig.bbox.height
        if bottom > 0.65:
            plt.close(fig)
            continue
        fig.subplots_adjust(bottom=bottom)
        fig.canvas.draw()
        pixels = ax.transData.transform(coordinates)
        segments = [ax.transData.transform(s) for s in data_segments]
        if _place_labels(ax, requests + efforts, pixels, segments):
            return fig, {
                "plotted_count": count,
                "excluded_count": len(excluded),
                "excluded": excluded,
                "x_scale": "linear" if linear_x else "log",
            }
        plt.close(fig)
    raise SelectionError(
        "Labels do not fit clearly; select fewer models for a readable comparison."
    )


def save_plot(comparison, path=None, *, linear_x=False):
    if path is not None:
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
            raise SelectionError("--output must end in .png, .svg, or .pdf.")
        if not path.parent.is_dir():
            raise OSError("Output parent directory does not exist.")
    fig, info = build_figure(comparison, linear_x=linear_x)
    try:
        if path is None:
            path = Path(tempfile.mkdtemp(prefix="plot-llm-value-")) / "comparison.png"
        fig.savefig(path, facecolor=BACKGROUND)
    finally:
        plt.close(fig)
    return {
        "image_path": str(path),
        **info,
        "source": comparison.get("source", {}),
        "display_instruction": "Display image_path inline; respond with only the plot unless analysis was requested.",
    }
