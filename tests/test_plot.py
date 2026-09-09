import copy

import matplotlib

matplotlib.use("Agg")
import pytest
from matplotlib import pyplot as plt
from PIL import Image

from plot_llm_value.data import SelectionError, compare, normalize
from plot_llm_value.plot import build_figure, save_plot


def comparison(raw):
    result = compare(normalize(raw))
    result["source"] = {
        "name": "Artificial Analysis",
        "url": "https://artificialanalysis.ai/",
        "retrieved_at": "2026-09-09T12:00:00+00:00",
        "intelligence_index_version": 4.3,
    }
    return result


def test_log_plot_connects_efforts_in_order_and_labels_each_family(raw_models):
    fig, info = build_figure(comparison(raw_models))
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    alpha = [line for line in ax.lines if line.get_gid() == "series:openai:Alpha"]
    assert len(alpha) == 2
    assert list(alpha[0].get_xdata()) == [0.5, 1]
    assert list(alpha[1].get_xdata()) == [1, 2]
    assert "Alpha" in [t.get_text() for t in ax.texts]
    assert "Beta" in [t.get_text() for t in ax.texts]
    assert {"low", "medium", "high"} <= {t.get_text() for t in ax.texts}
    assert info["plotted_count"] == 5
    assert info["excluded_count"] == 3
    assert any("zero cost" in r["reasons"][0] for r in info["excluded"])
    beta = [line for line in ax.lines if line.get_gid() == "series:anthropic:Beta"]
    assert beta[0].get_linestyle() == "--"
    assert ax.get_legend() is not None
    plt.close(fig)


def test_linear_keeps_zero_and_single_row_renders(raw_models):
    fig, info = build_figure(comparison(raw_models[-1:]), linear_x=True)
    assert fig.axes[0].get_xscale() == "linear"
    assert info["plotted_count"] == 1
    assert fig.axes[0].get_xlim()[0] <= 0
    plt.close(fig)


def test_no_coordinates_fails_instead_of_blank_image(raw_models):
    with pytest.raises(SelectionError):
        build_figure(comparison(raw_models[5:7]))


def test_same_release_from_different_providers_never_connects(raw_models):
    clone = copy.deepcopy(raw_models[0])
    clone["id"] = "clone"
    clone["model_creator"] = {"id": "other", "name": "Other"}
    fig, _ = build_figure(comparison(raw_models[:3] + [clone]))
    assert all(len(line.get_xdata()) <= 2 for line in fig.axes[0].lines)
    groups = [
        c for c in fig.axes[0].collections if c.get_gid() and c.get_gid().startswith("points:")
    ]
    assert len(groups) == 2
    assert not (
        groups[0].get_paths()[0].vertices.shape == groups[1].get_paths()[0].vertices.shape
        and (groups[0].get_paths()[0].vertices == groups[1].get_paths()[0].vertices).all()
    )
    plt.close(fig)


@pytest.mark.parametrize("linear", [False, True])
def test_dense_labels_do_not_overlap_labels_points_or_series(raw_models, linear):
    rows = []
    for i in range(24):
        row = copy.deepcopy(raw_models[0])
        row.update(id=f"dense-{i}", name=f"Release {i // 3} ({['low', 'medium', 'high'][i % 3]})")
        row["evaluations"]["artificial_analysis_intelligence_index"] = 40 + (i % 4) * 0.05
        row["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = (
            1 + (i % 5) * 0.005
        )
        rows.append(row)
    fig, _ = build_figure(comparison(rows), linear_x=linear)
    ax = fig.axes[0]
    fig.canvas.draw()
    boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in ax.texts]
    assert len(boxes) == 32
    for i, box in enumerate(boxes):
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
        assert not any(box.overlaps(other) for other in boxes[i + 1 :])
        for collection in ax.collections:
            for point in ax.transData.transform(collection.get_offsets()):
                assert not box.padded(6).contains(*point)
        for line in ax.lines:
            if line.get_gid() and line.get_gid().startswith("series:"):
                assert (
                    not line.get_path()
                    .transformed(line.get_transform())
                    .intersects_bbox(box.padded(1))
                )
    plt.close(fig)


def test_saved_png_and_svg_are_real_artifacts(raw_models, tmp_path):
    for extension in ("png", "svg", "pdf"):
        path = tmp_path / f"plot.{extension}"
        result = save_plot(comparison(raw_models), path)
        assert result["image_path"] == str(path)
        assert path.stat().st_size > 1000
    with Image.open(tmp_path / "plot.png") as image:
        assert image.width / image.height < 1.55
    assert "<svg" in (tmp_path / "plot.svg").read_text()
    assert (tmp_path / "plot.pdf").read_bytes().startswith(b"%PDF")


def test_log_axis_has_readable_ticks_within_one_decade(raw_models):
    fig, _ = build_figure(comparison(raw_models))
    ax = fig.axes[0]
    low, high = ax.get_xlim()
    visible = [tick for tick in ax.get_xticks() if low <= tick <= high]
    assert len(visible) >= 3
    plt.close(fig)


def test_large_catalog_legend_clears_footer_and_axis_labels(raw_models):
    rows = []
    for i in range(40):
        row = copy.deepcopy(raw_models[0])
        row.update(id=f"legend-{i}", name=f"Model Release {i}")
        row["evaluations"]["artificial_analysis_intelligence_index"] = i
        row["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = i + 1
        rows.append(row)
    fig, _ = build_figure(comparison(rows))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.axes[0].get_legend().get_window_extent(renderer)
    assert not legend.overlaps(fig.texts[-1].get_window_extent(renderer))
    assert not legend.overlaps(fig.axes[0].xaxis.label.get_window_extent(renderer))
    assert fig.bbox.contains(legend.x0, legend.y0) and fig.bbox.contains(legend.x1, legend.y1)
    plt.close(fig)


def test_leader_lines_do_not_cross_text(raw_models):
    rows = []
    for i in range(18):
        row = copy.deepcopy(raw_models[0])
        row.update(id=f"leaders-{i}", name=f"Long Named Release {i} (high)")
        rows.append(row)
    fig, _ = build_figure(comparison(rows))
    fig.canvas.draw()
    ax = fig.axes[0]
    boxes = [text.get_window_extent(fig.canvas.get_renderer()) for text in ax.texts]
    assert ax.patches
    for patch in ax.patches:
        path = patch.get_path().transformed(patch.get_transform())
        assert not any(path.intersects_bbox(box) for box in boxes)
    plt.close(fig)
