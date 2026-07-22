from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "mstyle_pretrained_centroid": "Pretrained mStyleDistance",
    "mstyle_finetuned_centroid": "Fine-tuned mStyleDistance",
    "mstyle_finetuned_prototype": "mStyleDistance + source prototypes",
    "challenger_pretrained_centroid": "Pretrained authorship representation",
    "challenger_finetuned_centroid": "Fine-tuned authorship representation",
    "classical_style": "Classical style-feature fusion",
    "learned_fusion": "Learned multi-view reranker",
}
METRICS = ("mrr", "recall_at_1", "recall_at_3", "recall_at_5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export publication-style method comparison artifacts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    metrics = report["test_metrics"]
    intervals = report.get("test_intervals", {})
    selected = report["decision"]
    protocol = report["protocol"]
    rows = []
    for method, values in metrics.items():
        row: dict[str, object] = {
            "method_id": method,
            "method": LABELS.get(method, method.replace("_", " ").title()),
            "selected": method == selected,
            "n_test_sources": protocol.get("n_test_sources", ""),
            "n_test_profiles": protocol.get("n_test_profiles", ""),
        }
        for metric in METRICS:
            row[metric] = values[metric]
            bounds = intervals.get(method, {}).get(metric, {})
            row[f"{metric}_ci_low"] = bounds.get("ci_low", "")
            row[f"{metric}_ci_high"] = bounds.get("ci_high", "")
        rows.append(row)
    rows.sort(key=lambda row: float(row["mrr"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["test_mrr_rank"] = rank

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "method_performance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out_dir / "method_performance.json"
    json_path.write_text(
        json.dumps({"protocol": protocol, "selected": selected, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, max(4.8, 0.52 * len(rows) + 1.5)),
        sharey=True,
    )
    colors = ["#7b2f3c" if row["selected"] else "#28476c" for row in rows]
    y = list(range(len(rows)))
    for axis, metric, title in zip(axes, ("mrr", "recall_at_3"), ("MRR", "Recall@3")):
        estimates = [float(row[metric]) for row in rows]
        lower = [float(row.get(f"{metric}_ci_low") or row[metric]) for row in rows]
        upper = [float(row.get(f"{metric}_ci_high") or row[metric]) for row in rows]
        errors = [
            [value - low for value, low in zip(estimates, lower)],
            [high - value for value, high in zip(estimates, upper)],
        ]
        axis.errorbar(estimates, y, xerr=errors, fmt="none", ecolor="#9d9485", elinewidth=1.2, capsize=2.5, zorder=1)
        axis.scatter(estimates, y, c=colors, s=48, zorder=2)
        for estimate, position in zip(estimates, y):
            axis.text(
                min(estimate + 0.012, axis.get_xlim()[1] - 0.04),
                position,
                f"{estimate:.3f}",
                va="center",
                fontsize=8.5,
                color="#3f3b35",
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("macro score · 95% profile-bootstrap CI")
        axis.set_xlim(0, min(1.0, max(0.4, max(upper) + 0.05)))
        axis.grid(axis="x", color="#ddd5c7", linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [str(row["method"]) for row in rows])
    axes[0].invert_yaxis()
    figure.suptitle("Source-heldout author retrieval", x=0.1, ha="left", fontsize=15, fontweight="bold")
    figure.patch.set_facecolor("#fffdf7")
    for axis in axes:
        axis.set_facecolor("#fffdf7")
    figure.tight_layout()
    png_path = args.out_dir / "method_performance.png"
    pdf_path = args.out_dir / "method_performance.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(pdf_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "png": str(png_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
