from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "customer_churn_dataset-training-master.csv").exists() else HERE.parent
TRAIN_FILE = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_FILE = DATA_DIR / "customer_churn_dataset-testing-master.csv"
OUTPUT = HERE / "Figure_2_1_research_framework.png"
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
print(f"Loaded Train ({len(train):,} rows) and Test ({len(test):,} rows).")

fig, ax = plt.subplots(figsize=(12, 8.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, text, *, fs=9, fc="#FFFFFF", bold=False):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.06",
                           facecolor=fc, edgecolor="#536273", linewidth=1.1)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", wrap=True)

def arrow(start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.1, color="#657789"))

def section(y, height, title):
    ax.add_patch(FancyBboxPatch((0.5, y), 11, height, boxstyle="round,pad=0.06",
                                facecolor="#F9FBFD", edgecolor="#A9B5C1", linewidth=1))
    ax.text(0.78, y+height-0.28, title, ha="left", va="center", fontsize=10,
            weight="bold", color="#25384A")

section(8.35, 1.15, "2.1 Data preprocessing")
prep = ["Data cleaning", "Missing-value\nimputation", "Categorical\nencoding", "Numeric\nstandardization"]
xs = [0.9, 3.55, 6.2, 8.85]
for i, label in enumerate(prep):
    box(xs[i], 8.55, 2.05, 0.55, label, fs=8.5)
    if i < 3:
        arrow((xs[i]+2.08, 8.825), (xs[i+1]-0.07, 8.825))
arrow((6, 8.32), (6, 7.88))

section(5.05, 3.0, "2.2 Churn prediction")
box(4.15, 7.35, 3.7, 0.48, "Stratified 5-fold cross-validation (one run)", fs=9, fc="#EAF0F6", bold=True)
for x, name in zip([1.05, 4.65, 8.25], ["Logistic Regression", "Random Forest", "XGBoost"]):
    box(x, 6.35, 2.7, 0.55, name, fs=9)
    arrow((6, 7.32), (x+1.35, 6.92))
box(3, 5.65, 6, 0.48, "Evaluation: Accuracy · Precision · Recall · F1-score · ROC-AUC", fs=8.6)
for x in [2.4, 6, 9.6]:
    arrow((x, 6.32), (6, 6.16))
box(3.65, 5.13, 4.7, 0.4, "Compare models; refit on all Train data; score Test", fs=8.7)
arrow((6, 5.62), (6, 5.55))

section(3.15, 1.4, "2.3 Customer value assessment")
box(3, 3.55, 6, 0.55,
    "Value score: Train-referenced percentiles of Spend, Usage Frequency and Tenure", fs=8.5)
arrow((6, 5.02), (6, 4.15))
section(1.2, 1.55, "2.4 Risk–value segmentation")
box(3.35, 1.55, 5.3, 0.58, "3 × 3 matrix: churn-risk tier × composite-value tier",
    fs=9, fc="#EAF0F6", bold=True)
arrow((6, 3.1), (6, 2.18))

ax.set_title("Research framework for churn prediction and risk–value prioritization",
             fontsize=14, weight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUTPUT, dpi=240, bbox_inches="tight")
plt.close(fig)
print("Saved:", OUTPUT)
