import time

import matplotlib.pyplot as plt
import pandas as pd

from autosklearn_utils import OUT, fit_outer_cv, load_datasets, summarize_scores

rows = []
for dataset, (X, y) in load_datasets().items():
    for minutes in (5, 15, 30):
        started = time.perf_counter()
        scores, _ = fit_outer_cv(dataset, X, y, seconds=minutes * 60, ensemble=True, seed_offset=1100 + minutes)
        elapsed = time.perf_counter() - started
        summary = summarize_scores(scores)
        rows.append({
            "Dataset": dataset,
            "Time budget per outer fold (min)": minutes,
            "F1-score mean": summary["F1 mean"],
            "F1-score SD": summary["F1 SD"],
            "Measured elapsed time (min)": elapsed / 60,
        })

data = pd.DataFrame(rows)
data.to_csv(OUT / "figure_3_1_data.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
for ax, dataset in zip(axes, ["Titanic", "Estonia"]):
    group = data[data["Dataset"] == dataset]
    ax2 = ax.twinx()
    score_line = ax.errorbar(group["Time budget per outer fold (min)"], group["F1-score mean"],
                             yerr=group["F1-score SD"], marker="o", capsize=4,
                             label="Outer-fold F1-score (mean ± SD)")
    runtime_line = ax2.plot(group["Time budget per outer fold (min)"], group["Measured elapsed time (min)"],
                            color="#ED7D31", marker="s", label="Measured total elapsed time")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax2.set_ylabel("Elapsed time (min)")
    ax.set_title(f"{dataset}: Auto-sklearn time-budget comparison")
    ax.grid(alpha=0.25)
    ax.legend([score_line, runtime_line[0]], [score_line.get_label(), runtime_line[0].get_label()], loc="best")
axes[-1].set_xlabel("Auto-sklearn search budget (min per outer fold)")
fig.tight_layout()
fig.savefig(OUT / "figure_3_1.png", dpi=200, bbox_inches="tight")
plt.close(fig)
