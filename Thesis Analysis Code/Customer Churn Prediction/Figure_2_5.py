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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "customer_churn_dataset-training-master.csv").exists() else HERE.parent
TRAIN_FILE = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_FILE = DATA_DIR / "customer_churn_dataset-testing-master.csv"
OUTPUT = HERE / "Figure_2_5_risk_value_matrix.png"
CSV_OUTPUT = HERE / "Figure_2_5_matrix_data.csv"

train = pd.read_csv(TRAIN_FILE).dropna(subset=["Churn"]).copy()
test = pd.read_csv(TEST_FILE).dropna(how="all").copy()
y = train["Churn"].astype(int)
X = train.drop(columns=["Churn", "CustomerID", "Contract Length"], errors="ignore")
X_test = test.drop(columns=["CustomerID", "Contract Length"], errors="ignore")

categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric = [c for c in X.columns if c not in categorical]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])
model = Pipeline([("prep", preprocessor),
                  ("model", LogisticRegression(max_iter=1200, class_weight="balanced"))])
model.fit(X, y)
test["RiskScore"] = model.predict_proba(X_test)[:, 1]

def train_percentile(train_values, values):
    reference = np.sort(np.asarray(train_values))
    return np.searchsorted(reference, np.asarray(values), side="right") / len(reference)

percentile_cols = []
for col in ["Total Spend", "Usage Frequency", "Tenure"]:
    pct_col = f"{col}_percentile"
    test[pct_col] = train_percentile(train[col], test[col])
    percentile_cols.append(pct_col)
test["CompositeValue"] = test[percentile_cols].mean(axis=1)

test["RiskTier"] = pd.qcut(test["RiskScore"].rank(method="first"), 3,
                           labels=["Low", "Medium", "High"])
test["ValueTier"] = pd.qcut(test["CompositeValue"].rank(method="first"), 3,
                            labels=["Low", "Medium", "High"])
test["PriorityIndex"] = test["RiskScore"] * test["CompositeValue"]

rows = []
for risk in ["Low", "Medium", "High"]:
    for value in ["Low", "Medium", "High"]:
        cell = test[(test.RiskTier == risk) & (test.ValueTier == value)]
        rows.append({
            "risk_tier": risk,
            "value_tier": value,
            "n": len(cell),
            "share_of_test": len(cell) / len(test),
            "mean_predicted_churn_score": cell["RiskScore"].mean(),
            "mean_composite_value": cell["CompositeValue"].mean(),
            "mean_priority_index": cell["PriorityIndex"].mean(),
        })
matrix_data = pd.DataFrame(rows)
matrix_data.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

risk_order = ["High", "Medium", "Low"]
value_order = ["Low", "Medium", "High"]
priority = matrix_data.pivot(index="risk_tier", columns="value_tier",
                             values="mean_priority_index").reindex(
                                 index=risk_order, columns=value_order)
fig, ax = plt.subplots(figsize=(10.5, 7.2))
image = ax.imshow(priority.to_numpy(), cmap="YlOrRd", vmin=0,
                   vmax=float(priority.to_numpy().max()), aspect="auto")
for i, risk in enumerate(risk_order):
    for j, value in enumerate(value_order):
        row = matrix_data[(matrix_data.risk_tier == risk) &
                          (matrix_data.value_tier == value)].iloc[0]
        ax.text(j, i,
                f"Customers: {int(row.n):,} ({row.share_of_test:.1%})\n"
                f"Mean risk: {row.mean_predicted_churn_score:.1%}\n"
                f"Mean composite value: {row.mean_composite_value:.1%}\n"
                f"Mean priority index: {row.mean_priority_index:.3f}",
                ha="center", va="center", fontsize=8.5,
                color="white" if row.mean_priority_index > priority.to_numpy().max()*0.58 else "#222222")
ax.set_xticks(range(3), ["Low value", "Medium value", "High value"])
ax.set_yticks(range(3), ["High relative risk", "Medium relative risk", "Low relative risk"])
ax.set_xlabel("Value tier (equal-weight composite value tertiles within Test)")
ax.set_ylabel("Risk tier (Logistic Regression score tertiles within Test)")
ax.set_title("Customer risk–value priority matrix\n"
             "Value = mean of Train-referenced Spend, Usage Frequency and Tenure percentiles")
colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
colorbar.set_label("Mean priority index (predicted churn score × composite value)")
fig.text(0.5, 0.012, "Risk and value tiers are cohort-relative; Test has no observed churn labels.",
         ha="center", fontsize=8.5, color="#444444")
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig(OUTPUT, dpi=240, bbox_inches="tight"); plt.close(fig)
print(f"Loaded Train ({len(train):,}) and Test ({len(test):,}) rows.")
print("Saved:", OUTPUT)
print("Matrix data saved:", CSV_OUTPUT)
