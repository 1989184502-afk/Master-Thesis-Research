from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "customer_churn_dataset-training-master.csv").exists() else HERE.parent
TRAIN_FILE = DATA_DIR / "customer_churn_dataset-training-master.csv"
OUTPUT = HERE / "Table_2_1_cv_metrics.png"
CSV_OUTPUT = HERE / "Table_2_1_cv_metrics.csv"

train = pd.read_csv(TRAIN_FILE).dropna(subset=["Churn"]).copy()
y = train["Churn"].astype(int)
X = train.drop(columns=["Churn", "CustomerID", "Contract Length"], errors="ignore")

def make_pipeline(model):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("prep", prep), ("model", model)])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss", n_jobs=1,
        random_state=42, tree_method="hist"),
}
rows = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(cv.split(X, y), 1):
    print(f"Fold {fold}/5")
    for name, model in models.items():
        pipe = make_pipeline(model)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[va])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({"Model": name, "fold": fold,
            "Accuracy": accuracy_score(y.iloc[va], pred),
            "Precision": precision_score(y.iloc[va], pred, zero_division=0),
            "Recall": recall_score(y.iloc[va], pred, zero_division=0),
            "F1-score": f1_score(y.iloc[va], pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y.iloc[va], prob)})

fold_results = pd.DataFrame(rows)
metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
table = []
for name in models:
    part = fold_results[fold_results["Model"] == name]
    table.append({"Model": name, **{
        m: f"{part[m].mean():.4f} ± {part[m].std(ddof=1):.4f}" for m in metrics}})
result = pd.DataFrame(table)
result.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(12, 2.5)); ax.axis("off")
t = ax.table(cellText=result.values, colLabels=result.columns, cellLoc="center", loc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.65)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#D7DDE5")
    if r == 0:
        cell.set_facecolor("#EAF0F6"); cell.set_text_props(weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F7F9FB")
ax.set_title("Table 2.1. Comparative performance of churn prediction models", pad=14, weight="bold")
fig.tight_layout(); fig.savefig(OUTPUT, dpi=240, bbox_inches="tight"); plt.close(fig)
print("Saved:", OUTPUT)
print(result.to_string(index=False))
