import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, roc_auc_score

from autosklearn_utils import OUT, load_datasets, outer_prediction_cv

restricted_pool = {"classifier": ["decision_tree", "random_forest", "libsvm_svc"]}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
rows = []
for ax, (dataset, (X, y)) in zip(axes, load_datasets().items()):
    for label, include, offset in [
        ("Unrestricted Auto-sklearn", None, 1400),
        ("Restricted Auto-sklearn", restricted_pool, 1500),
    ]:
        actual, probability = outer_prediction_cv(dataset, X, y, seconds=900, include=include, seed_offset=offset)
        auc = roc_auc_score(actual, probability)
        RocCurveDisplay.from_predictions(actual, probability, ax=ax, name=f"{label} (AUC={auc:.3f})")
        rows.extend({
            "Dataset": dataset,
            "Search space": label,
            "Actual": int(target),
            "Predicted survival probability": float(prob),
        } for target, prob in zip(actual, probability))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title(f"{dataset}: outer-fold ROC curves")
    ax.legend(loc="lower right", fontsize=8)
data = pd.DataFrame(rows)
data.to_csv(OUT / "figure_3_3_oof_predictions.csv", index=False, encoding="utf-8-sig")
fig.tight_layout()
fig.savefig(OUT / "figure_3_3.png", dpi=200, bbox_inches="tight")
plt.close(fig)
