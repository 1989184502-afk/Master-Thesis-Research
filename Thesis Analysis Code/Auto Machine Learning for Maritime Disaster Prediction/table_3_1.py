import pandas as pd
from autosklearn_utils import load_datasets, save_table

rows = []
for dataset, (X, y) in load_datasets().items():
    rows.append({
        "Dataset": dataset,
        "Observations": len(y),
        "Survivors (n)": int(y.sum()),
        "Survival rate": f"{y.mean():.1%}",
        "Predictors": X.shape[1],
        "Predictor fields": ", ".join(X.columns.astype(str)),
    })
save_table(pd.DataFrame(rows), "table_3_1_dataset_profile")
