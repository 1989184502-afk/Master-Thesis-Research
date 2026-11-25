from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "customer_churn_dataset-training-master.csv").exists() else HERE.parent
TRAIN_FILE = DATA_DIR / "customer_churn_dataset-training-master.csv"
OUTPUT = HERE / "Figure_2_3_mean_roc_curves.png"
train = pd.read_csv(TRAIN_FILE).dropna(subset=["Churn"]).copy()
y = train["Churn"].astype(int)
X = train.drop(columns=["Churn", "CustomerID", "Contract Length"], errors="ignore")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss", n_jobs=1,
        random_state=42, tree_method="hist"),
}
grid = np.linspace(0, 1, 501)
curves = {name: [] for name in models}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr, va) in enumerate(cv.split(X, y), 1):
    print(f"Fold {fold}/5")
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    categorical = X_tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X_tr.columns if c not in categorical]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    for name, model in models.items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(X_tr, y.iloc[tr])
        prob = pipe.predict_proba(X_va)[:, 1]
        fpr, tpr, _ = roc_curve(y.iloc[va], prob)
        interp = np.interp(grid, fpr, tpr)
        interp[0], interp[-1] = 0, 1
        curves[name].append(interp)

fig, ax = plt.subplots(figsize=(8.2, 6.6))
colors = {"Logistic Regression": "#2878B5", "Random Forest": "#F08A24", "XGBoost": "#3A9D5D"}
for name, runs in curves.items():
    array = np.asarray(runs)
    mean_tpr = array.mean(axis=0)
    sd_tpr = array.std(axis=0, ddof=1)
    mean_auc = np.trapz(mean_tpr, grid)
    ax.plot(grid, mean_tpr, lw=2.3, color=colors[name], label=f"{name} (mean AUC={mean_auc:.3f})")
    ax.fill_between(grid, np.maximum(mean_tpr-sd_tpr, 0), np.minimum(mean_tpr+sd_tpr, 1),
                    color=colors[name], alpha=0.14)
ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1, label="Chance")
ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="Mean five-fold ROC curves with ±1 SD bands")
ax.grid(alpha=0.2); ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUTPUT, dpi=240, bbox_inches="tight"); plt.close(fig)
print("Saved:", OUTPUT)
