import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from autosklearn_utils import load_datasets, fit_outer_cv, evaluate_baseline, summarize_scores, save_table

rows = []
baselines = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=140, max_depth=3, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.9, reg_lambda=1, objective="binary:logistic",
                              eval_metric="logloss", n_jobs=1, random_state=42, verbosity=0),
}
for dataset, (X, y) in load_datasets().items():
    auto_scores, _ = fit_outer_cv(dataset, X, y, seconds=900, ensemble=True, seed_offset=600)
    all_scores = [("Auto-sklearn (unrestricted, 15 min/fold)", auto_scores)]
    for name, estimator in baselines.items():
        all_scores.append((name, evaluate_baseline(estimator, X, y)))
    for name, scores in all_scores:
        summary = summarize_scores(scores)
        rows.append({
            "Dataset": dataset,
            "Model": name,
            "F1-score (mean ± SD)": f"{summary['F1 mean']:.3f} ± {summary['F1 SD']:.3f}",
            "ROC-AUC (mean ± SD)": f"{summary['ROC-AUC mean']:.3f} ± {summary['ROC-AUC SD']:.3f}",
            "Accuracy (mean ± SD)": f"{summary['Accuracy mean']:.3f} ± {summary['Accuracy SD']:.3f}",
        })
save_table(pd.DataFrame(rows), "table_3_4_automl_vs_conventional")
