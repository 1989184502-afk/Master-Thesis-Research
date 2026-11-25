import pandas as pd
from autosklearn_utils import load_datasets, fit_outer_cv, summarize_scores, save_table

rows = []
for dataset, (X, y) in load_datasets().items():
    scores, models = fit_outer_cv(dataset, X, y, seconds=900, ensemble=True, seed_offset=900)
    summary = summarize_scores(scores)
    rows.append({
        "Dataset": dataset,
        "Imbalance treatment": "Auto-sklearn native balancing strategy searched automatically",
        "Selected estimator(s) across outer folds": "; ".join(sorted(set(models))),
        "F1-score (mean ± SD)": f"{summary['F1 mean']:.3f} ± {summary['F1 SD']:.3f}",
        "ROC-AUC (mean ± SD)": f"{summary['ROC-AUC mean']:.3f} ± {summary['ROC-AUC SD']:.3f}",
        "Accuracy (mean ± SD)": f"{summary['Accuracy mean']:.3f} ± {summary['Accuracy SD']:.3f}",
        "Recall (mean ± SD)": f"{summary['Recall mean']:.3f} ± {summary['Recall SD']:.3f}",
    })
save_table(pd.DataFrame(rows), "table_3_6_native_imbalance_handling")
