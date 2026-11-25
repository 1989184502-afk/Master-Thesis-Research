import pandas as pd
from autosklearn_utils import load_datasets, fit_outer_cv, summarize_scores, candidate_scores, save_table

rows = []
top_rows = []
restricted_pool = {"classifier": ["decision_tree", "random_forest", "libsvm_svc"]}
for dataset, (X, y) in load_datasets().items():
    for label, include, offset in [
        ("Unrestricted Auto-sklearn search", None, 300),
        ("Restricted search: decision tree, random forest, SVM", restricted_pool, 400),
    ]:
        scores, models = fit_outer_cv(dataset, X, y, seconds=900, include=include, seed_offset=offset)
        summary = summarize_scores(scores)
        rows.append({
            "Dataset": dataset,
            "Search space": label,
            "Selected estimator(s) across outer folds": "; ".join(sorted(set(models))),
            "F1-score (mean ± SD)": f"{summary['F1 mean']:.3f} ± {summary['F1 SD']:.3f}",
            "ROC-AUC (mean ± SD)": f"{summary['ROC-AUC mean']:.3f} ± {summary['ROC-AUC SD']:.3f}",
            "Accuracy (mean ± SD)": f"{summary['Accuracy mean']:.3f} ± {summary['Accuracy SD']:.3f}",
        })
    candidates = candidate_scores(dataset, X, y, seconds=900, seed_offset=500)
    if candidates:
        frame = pd.DataFrame(candidates)
        ranked = frame.groupby("Rank", as_index=False).agg(
            **{"Internal validation F1-score": ("Validation F1-score", "mean"), "Outer-fold appearances": ("Outer fold", "nunique"), "Model IDs": ("Model ID", lambda values: ", ".join(values.astype(str)))}
        ).sort_values("Rank").head(5)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            top_rows.append({
                "Dataset": dataset,
                "Rank": rank,
                "Auto-sklearn candidate": f"Rank {rank} pipeline in fold-specific leaderboard",
                "Model IDs by outer fold": row["Model IDs"],
                "Outer-fold appearances": int(row["Outer-fold appearances"]),
                "Internal validation F1-score": f"{row['Internal validation F1-score']:.3f}",
            })
save_table(pd.DataFrame(rows), "table_3_3_search_space_comparison")
if top_rows:
    save_table(pd.DataFrame(top_rows), "table_3_3a_top_five_candidates")
