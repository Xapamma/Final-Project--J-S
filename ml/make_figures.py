"""Regenerate the result figures in results/media/ from the saved score tables.

Usage:
    uv run python -m ml.make_figures

Reads results/tables/tuned_comparison_nov19_scores.csv and
results/tables/feature_ablation_scores.csv; writes accuracy_comparison.svg,
macro_f1_comparison.svg and feature_amount_effect.svg.
"""

import argparse
from pathlib import Path

import pandas as pd

METHOD_LABELS = {
    "most_common": "Most common",
    "knn_tfidf": "TF-IDF KNN",
    "knn_embeddings": "Embedding KNN",
    "random_forest": "Random forest",
    "neural_network": "Neural network",
}
PEOPLE = ["Savanna", "Jillian"]
COLORS = {"Savanna": "#0072B2", "Jillian": "#E69F00"}


def bar_chart(scores, metric, title):
    """Grouped bar chart (one group per method, one bar per person)."""
    top, bottom, left, right = 70, 390, 80, 960
    height = bottom - top
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="540" '
        f'viewBox="0 0 1000 540" role="img" aria-label="{title}">',
        '<rect width="1000" height="540" fill="white" />',
        '<g font-family="Arial,sans-serif" fill="#222">',
        f'<text x="500" y="30" font-size="22" text-anchor="middle" font-weight="bold">{title}</text>',
    ]
    for i in range(6):
        y = bottom - height * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.0f}" x2="{right}" y2="{y:.0f}" stroke="#ddd" />')
        parts.append(f'<text x="68" y="{y + 5:.0f}" font-size="13" text-anchor="end">{i / 5:.1f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#555" />')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#555" />')
    for k, (method, label) in enumerate(METHOD_LABELS.items()):
        center = 166 + 176 * k
        for j, person in enumerate(PEOPLE):
            value = scores.loc[(scores.person == person) & (scores.method == method), metric].iloc[0]
            # 38px bars, 6px apart: label centers sit 44px apart, wider than a 3-decimal label
            x = center - 41 + 44 * j
            h = value * height
            parts.append(f'<rect x="{x}" y="{bottom - h:.2f}" width="38" height="{h:.2f}" fill="{COLORS[person]}" />')
            parts.append(f'<text x="{x + 19}" y="{bottom - h - 7:.2f}" font-size="12" '
                         f'text-anchor="middle">{value:.3f}</text>')
        parts.append(f'<text x="{center}" y="415" font-size="13" transform="rotate(-35 {center} 415)" '
                     f'text-anchor="end">{label}</text>')
    parts.append('<rect x="390" y="500" width="14" height="14" fill="#0072B2" />'
                 '<text x="412" y="512" font-size="14">Savanna</text>')
    parts.append('<rect x="530" y="500" width="14" height="14" fill="#E69F00" />'
                 '<text x="552" y="512" font-size="14">Jillian</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def amount_effect_chart(ablation):
    """Horizontal diverging bars: within-person macro F1 change from adding amount."""
    pooled = ablation[(ablation.setting == "within") & (ablation.fold == "pooled")]
    zero, scale = 465, 1700  # pixels per unit of macro F1
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="410" viewBox="0 0 850 410">',
        '<rect width="850" height="410" fill="white" />',
        '<g font-family="Arial,sans-serif" fill="#172033">',
        '<text x="20" y="30" font-size="21" font-weight="bold">'
        'Adding amount: change in within-person macro F1</text>',
        '<text x="20" y="53" font-size="14">Compared with description alone; '
        'fixed model settings and the same held-out folds</text>',
        f'<line x1="{zero}" y1="75" x2="{zero}" y2="360" stroke="#4a5568" />',
    ]
    row = 0
    for person in PEOPLE:
        for method in ["knn_tfidf", "knn_embeddings", "random_forest", "neural_network"]:
            sub = pooled[(pooled.train_person == person) & (pooled.method == method)]
            base = sub.loc[sub.feature_variant == "description", "macro_f1"].iloc[0]
            delta = sub.loc[sub.feature_variant == "description + amount", "macro_f1"].iloc[0] - base
            y = 82 + 34 * row
            width = abs(delta) * scale
            parts.append(f'<text x="20" y="{y + 17}" font-size="15">{person} · {METHOD_LABELS[method]}</text>')
            if delta >= 0:
                parts.append(f'<rect x="{zero:.1f}" y="{y}" width="{width:.1f}" height="22" fill="#23877e" />')
                parts.append(f'<text x="{zero + width + 10:.1f}" y="{y + 17}" font-size="15" '
                             f'text-anchor="start">{delta:+.3f}</text>')
            else:
                parts.append(f'<rect x="{zero - width:.1f}" y="{y}" width="{width:.1f}" height="22" fill="#bc6543" />')
                parts.append(f'<text x="{zero - width - 10:.1f}" y="{y + 17}" font-size="15" '
                             f'text-anchor="end">{delta:+.3f}</text>')
            row += 1
    parts.append(f'<text x="{zero}" y="385" text-anchor="middle" font-size="14">0 = no change</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="results/tables/tuned_comparison_nov19_scores.csv")
    parser.add_argument("--ablation", default="results/tables/feature_ablation_scores.csv")
    parser.add_argument("--media", default="results/media")
    args = parser.parse_args()
    media = Path(args.media)
    media.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    ablation = pd.read_csv(args.ablation, dtype={"fold": str})
    outputs = {
        "accuracy_comparison.svg": bar_chart(scores, "accuracy", "Within-person accuracy"),
        "macro_f1_comparison.svg": bar_chart(scores, "macro_f1", "Within-person macro F1"),
        "feature_amount_effect.svg": amount_effect_chart(ablation),
    }
    for name, svg in outputs.items():
        (media / name).write_text(svg, encoding="utf-8")
        print(f"Wrote {media / name}")


if __name__ == "__main__":
    main()
