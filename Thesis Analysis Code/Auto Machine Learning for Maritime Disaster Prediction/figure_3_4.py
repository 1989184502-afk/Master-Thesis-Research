import matplotlib.pyplot as plt
import pandas as pd

from autosklearn_utils import OUT, fit_outer_cv, load_datasets, summarize_scores

rows = []
for dataset, (X, y) in load_datasets().items():
    scores, _ = fit_outer_cv(dataset, X, y, seconds=900, ensemble=True, seed_offset=1600)
    summary = summarize_scores(scores)
    for metric in ("F1", "ROC-AUC", "Accuracy", "Precision", "Recall"):
        rows.append({
            "Dataset": dataset,
            "Metric": metric,
            "Mean": summary[f"{metric} mean"],
            "SD": summary[f"{metric} SD"],
        })

data = pd.DataFrame(rows)
data.to_csv(OUT / "figure_3_4_data.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
colors = ["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5", "#8064A2"]
for ax, dataset in zip(axes, ["Titanic", "Estonia"]):
    group = data[data["Dataset"] == dataset]
    ax.bar(group["Metric"], group["Mean"], yerr=group["SD"], capsize=3, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Outer-fold score (mean ± SD)")
    ax.set_title(f"{dataset}: Auto-sklearn performance profile")
    ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(OUT / "figure_3_4.png", dpi=200, bbox_inches="tight")
plt.close(fig)
