from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "customer_churn_dataset-training-master.csv").exists() else HERE.parent
TRAIN_FILE = DATA_DIR / "customer_churn_dataset-training-master.csv"
OUTPUT = HERE / "Figure_2_2_numeric_by_churn.png"
train = pd.read_csv(TRAIN_FILE).dropna(subset=["Churn"]).copy()
train["Churn"] = train["Churn"].astype(int)
columns = ["Age", "Tenure", "Usage Frequency", "Support Calls",
           "Payment Delay", "Total Spend", "Last Interaction"]
colors = {0: "#4C78A8", 1: "#E45756"}

fig, axes = plt.subplots(2, 4, figsize=(15, 7.3))
for ax, col in zip(axes.flat, columns):
    groups = [train.loc[train.Churn == label, col].dropna().to_numpy() for label in (0, 1)]
    bp = ax.boxplot(groups, labels=["No", "Yes"], showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.3})
    for patch, label in zip(bp["boxes"], (0, 1)):
        patch.set_facecolor(colors[label]); patch.set_alpha(0.72)
    ax.set_title(col); ax.set_xlabel("Churn"); ax.grid(axis="y", alpha=0.2)
axes.flat[-1].axis("off")
axes[0, 0].set_ylabel("Value"); axes[1, 0].set_ylabel("Value")
fig.suptitle("Numeric customer characteristics by churn status (outliers hidden)",
             y=1.01, fontsize=13)
fig.tight_layout(); fig.savefig(OUTPUT, dpi=240, bbox_inches="tight"); plt.close(fig)
print(f"Loaded {len(train):,} labeled Train records.")
print("Saved:", OUTPUT)
