import pandas as pd
from autosklearn_utils import load_datasets, fit_outer_cv, summarize_scores, save_table

rows = []
for dataset, (X, y) in load_datasets().items():
    for minutes in (5, 15, 30):
        scores, _ = fit_outer_cv(dataset, X, y, seconds=minutes * 60, ensemble=True, seed_offset=minutes)
        summary = summarize_scores(scores)
        rows.append({
            "Dataset": dataset,
            "Auto-sklearn time budget (min/fold)": minutes,
            "Search space": "Unrestricted Auto-sklearn search",
            "Outer evaluation": "Stratified 5-fold CV",
            "F1-score (mean ± SD)": f"{summary['F1 mean']:.3f} ± {summary['F1 SD']:.3f}",
            "ROC-AUC (mean ± SD)": f"{summary['ROC-AUC mean']:.3f} ± {summary['ROC-AUC SD']:.3f}",
            "Accuracy (mean ± SD)": f"{summary['Accuracy mean']:.3f} ± {summary['Accuracy SD']:.3f}",
        })
save_table(pd.DataFrame(rows), "table_3_2_time_budget")
