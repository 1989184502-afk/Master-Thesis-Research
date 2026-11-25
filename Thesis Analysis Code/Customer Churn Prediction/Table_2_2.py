from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "customer_churn_dataset-training-master.csv").exists() else HERE.parent
TRAIN_FILE = DATA_DIR / "customer_churn_dataset-training-master.csv"
OUTPUT = HERE / "Table_2_2_value_risk_metrics.png"
CSV_OUTPUT = HERE / "Table_2_2_value_risk_metrics.csv"

train = pd.read_csv(TRAIN_FILE).dropna(subset=["Churn"]).copy()
y = train["Churn"].astype(int)
X = train.drop(columns=["Churn", "CustomerID", "Contract Length"], errors="ignore")
baseline_oof = np.zeros(len(train)); direct_oof = np.zeros(len(train)); z_oof = np.zeros(len(train))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_preprocessor(data):
    categorical = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in data.columns if c not in categorical]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

for fold, (tr, va) in enumerate(cv.split(X, y), 1):
    print(f"Fold {fold}/5")
    reference = np.sort(train.iloc[tr]["Total Spend"].to_numpy())
    v_tr = np.searchsorted(reference, train.iloc[tr]["Total Spend"].to_numpy(), side="right") / len(reference)
    v_va = np.searchsorted(reference, train.iloc[va]["Total Spend"].to_numpy(), side="right") / len(reference)
    z_tr = y.iloc[tr].to_numpy() * v_tr
    z_va = y.iloc[va].to_numpy() * v_va
    z_oof[va] = z_va

    classifier = Pipeline([("prep", make_preprocessor(X.iloc[tr])),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=1, random_state=42))])
    classifier.fit(X.iloc[tr], y.iloc[tr])
    baseline_oof[va] = classifier.predict_proba(X.iloc[va])[:, 1] * v_va

    regressor = Pipeline([("prep", make_preprocessor(X.iloc[tr])),
        ("model", RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
            max_features=0.8, n_jobs=1, random_state=42))])
    regressor.fit(X.iloc[tr], z_tr)
    direct_oof[va] = np.clip(regressor.predict(X.iloc[va]), 0, 1)

def score_row(label, prediction):
    return {"Method": label,
        "MAE": mean_absolute_error(z_oof, prediction),
        "RMSE": mean_squared_error(z_oof, prediction) ** 0.5,
        "R²": r2_score(z_oof, prediction),
        "Spearman correlation with observed Z": pd.Series(z_oof).corr(pd.Series(prediction), method="spearman")}

result = pd.DataFrame([
    score_row("RF churn score × spending percentile (OOF baseline)", baseline_oof),
    score_row("Random Forest Regressor predicting Z directly", direct_oof),
])
result.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
shown = result.copy()
for col in ["MAE", "RMSE", "R²", "Spearman correlation with observed Z"]:
    shown[col] = shown[col].map(lambda v: f"{v:.5f}")
fig, ax = plt.subplots(figsize=(12, 2.6)); ax.axis("off")
t = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", loc="center")
t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.75)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#D7DDE5")
    if r == 0:
        cell.set_facecolor("#EAF0F6"); cell.set_text_props(weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F7F9FB")
ax.set_title("Table 2.2. Performance metrics for direct churn-weighted value prediction", pad=14, weight="bold")
fig.tight_layout(); fig.savefig(OUTPUT, dpi=240, bbox_inches="tight"); plt.close(fig)
print("Saved:", OUTPUT)
print(result.round(5).to_string(index=False))
