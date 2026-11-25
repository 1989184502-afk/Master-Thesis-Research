import matplotlib.pyplot as plt
import pandas as pd

from autosklearn_utils import OUT, fit_outer_cv, load_datasets, summarize_scores

restricted_pool = {"classifier": ["decision_tree", "random_forest", "libsvm_svc"]}
rows = []
for dataset, (X, y) in load_datasets().items():
    for label, include, offset in [
        ("Unrestricted search", None, 1200),
        ("Restricted search (three classifiers)", restricted_pool, 1300),
    ]:
        scores, _ = fit_outer_cv(dataset, X, y, seconds=900, include=include, seed_offset=offset)
        summary = summarize_scores(scores)
        rows.append({
            "Dataset": dataset,
            "Search space": label,
            "F1-score mean": summary["F1 mean"],
            "F1-score SD": summary["F1 SD"],
            "ROC-AUC mean": summary["ROC-AUC mean"],
            "ROC-AUC SD": summary["ROC-AUC SD"],
        })

data = pd.DataFrame(rows)
data.to_csv(OUT / "figure_3_2_data.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, dataset in zip(axes, ["Titanic", "Estonia"]):
    group = data[data["Dataset"] == dataset]
    x = range(len(group))
    ax.bar(x, group["F1-score mean"], yerr=group["F1-score SD"], capsize=4,
           color=["#4472C4", "#ED7D31"])
    ax.set_xticks(list(x), ["Unrestricted", "Restricted"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Outer-fold F1-score (mean ± SD)")
    ax.set_title(f"{dataset}: search-space comparison")
    for position, (_, row) in enumerate(group.iterrows()):
        ax.text(position, min(row["F1-score mean"] + row["F1-score SD"] + 0.025, 0.97),
                f"AUC {row['ROC-AUC mean']:.3f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figure_3_2.png", dpi=200, bbox_inches="tight")
plt.close(fig)
