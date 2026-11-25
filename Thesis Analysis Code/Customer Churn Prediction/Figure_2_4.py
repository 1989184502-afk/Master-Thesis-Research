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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "customer_churn_dataset-training-master.csv").exists() else HERE.parent
TRAIN_FILE = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_FILE = DATA_DIR / "customer_churn_dataset-testing-master.csv"
OUTPUT = HERE / "Figure_2_4_test_score_distributions.png"
train = pd.read_csv(TRAIN_FILE).dropna(subset=["Churn"]).copy()
test = pd.read_csv(TEST_FILE).dropna(how="all").copy()
y = train["Churn"].astype(int)
X = train.drop(columns=["Churn", "CustomerID", "Contract Length"], errors="ignore")
X_test = test.drop(columns=["CustomerID", "Contract Length"], errors="ignore")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss", n_jobs=1,
        random_state=42, tree_method="hist"),
}
categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric = [c for c in X.columns if c not in categorical]
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

fig, ax = plt.subplots(figsize=(9, 6))
colors = {"Logistic Regression": "#2878B5", "Random Forest": "#F08A24", "XGBoost": "#3A9D5D"}
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X, y)
    score = pipe.predict_proba(X_test)[:, 1]
    ax.hist(score, bins=35, density=True, histtype="step", lw=2,
            color=colors[name], label=name)
ax.axvline(0.5, color="#555555", linestyle="--", lw=1, label="Classification threshold = 0.5")
ax.set(xlabel="Predicted churn score", ylabel="Density",
       title="Predicted churn-score distributions in the Test set")
ax.legend(); ax.grid(axis="y", alpha=0.2)
fig.tight_layout(); fig.savefig(OUTPUT, dpi=240, bbox_inches="tight"); plt.close(fig)
print(f"Loaded {len(train):,} Train rows and {len(test):,} Test rows.")
print("Saved:", OUTPUT)
