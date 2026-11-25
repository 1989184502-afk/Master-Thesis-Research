from pathlib import Path
import os
import re
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
OUTER_FOLDS = 5


def load_datasets():
    titanic = pd.read_csv(ROOT / "train_and_test2.csv")
    titanic = titanic.loc[titanic["Passengerid"] <= 891].copy()
    titanic = titanic.loc[:, titanic.nunique(dropna=False) > 1]
    y_titanic = titanic.pop("2urvived").astype(int)
    x_titanic = titanic.drop(columns=["Passengerid"], errors="ignore")

    estonia = pd.read_csv(ROOT / "estonia-passenger-list.csv")
    y_estonia = estonia.pop("Survived").astype(int)
    x_estonia = estonia.drop(columns=["PassengerId", "Firstname", "Lastname"], errors="ignore")
    return {"Titanic": (x_titanic, y_titanic), "Estonia": (x_estonia, y_estonia)}


def autosklearn_imports():
    try:
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn.metrics import f1
    except Exception as exc:
        raise RuntimeError(
            "Auto-sklearn could not be imported. Run this project in Linux/WSL2 or a Linux container "
            "with the packages in requirements.txt installed. Native Windows is not supported."
        ) from exc
    return AutoSklearnClassifier, f1


def make_automl(seconds, dataset, fold, ensemble=True, include=None, exclude=None, seed_offset=0):
    AutoSklearnClassifier, f1 = autosklearn_imports()
    temp_path = Path(tempfile.mkdtemp(prefix=f"ask_{re.sub('[^a-z0-9]+','_',dataset.lower())}_"))
    return AutoSklearnClassifier(
        time_left_for_this_task=int(seconds),
        per_run_time_limit=max(30, min(300, int(seconds) // 4)),
        metric=f1,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": 5, "shuffle": True},
        seed=SEED + fold + seed_offset,
        memory_limit=4096,
        include=include,
        exclude=exclude,
        ensemble_class="default" if ensemble else None,
        ensemble_size=50 if ensemble else 0,
        max_models_on_disc=100,
        n_jobs=1,
        tmp_folder=str(temp_path),
        delete_tmp_folder_after_terminate=False,
        allow_string_features=True,
    ), temp_path


def outer_splits(X, y):
    return list(StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED).split(X, y))


def score_predictions(y_true, prediction, probability):
    return {
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
    }


def summarize_scores(scores):
    frame = pd.DataFrame(scores)
    result = {}
    for col in frame.columns:
        result[f"{col} mean"] = float(frame[col].mean())
        result[f"{col} SD"] = float(frame[col].std(ddof=1))
    result["Successful folds"] = f"{len(frame)}/{OUTER_FOLDS}"
    return result


def fit_outer_cv(dataset, X, y, seconds=900, ensemble=True, include=None, exclude=None, seed_offset=0):
    folds = []
    model_names = []
    for fold, (train_idx, test_idx) in enumerate(outer_splits(X, y), start=1):
        automl, temp_path = make_automl(seconds, dataset, fold, ensemble, include, exclude, seed_offset)
        try:
            automl.fit(X.iloc[train_idx], y.iloc[train_idx])
            automl.refit(X.iloc[train_idx], y.iloc[train_idx])
            probability = automl.predict_proba(X.iloc[test_idx])[:, 1]
            prediction = automl.predict(X.iloc[test_idx])
            folds.append(score_predictions(y.iloc[test_idx], prediction, probability))
            model_names.append(selected_model_name(automl))
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)
    return folds, model_names


def outer_prediction_cv(dataset, X, y, seconds=900, include=None, seed_offset=0):
    actual = np.zeros(len(y), dtype=int)
    probability = np.full(len(y), np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(outer_splits(X, y), start=1):
        automl, temp_path = make_automl(seconds, dataset, fold, True, include, None, seed_offset)
        try:
            automl.fit(X.iloc[train_idx], y.iloc[train_idx])
            automl.refit(X.iloc[train_idx], y.iloc[train_idx])
            actual[test_idx] = y.iloc[test_idx].to_numpy()
            probability[test_idx] = automl.predict_proba(X.iloc[test_idx])[:, 1]
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)
    return actual, probability


def selected_model_name(automl):
    try:
        models = automl.show_models()
        names = []
        for model in models.values():
            classifier = model.get("classifier")
            choice = getattr(classifier, "choice", None) if classifier is not None else None
            names.append(type(choice).__name__ if choice is not None else str(classifier))
        return " + ".join(names) if names else "Auto-sklearn ensemble"
    except Exception:
        return "Auto-sklearn ensemble"


def candidate_scores(dataset, X, y, seconds=900, include=None, seed_offset=0):
    records = []
    for fold, (train_idx, test_idx) in enumerate(outer_splits(X, y), start=1):
        automl, temp_path = make_automl(seconds, dataset, fold, True, include, None, seed_offset)
        try:
            automl.fit(X.iloc[train_idx], y.iloc[train_idx])
            leaderboard = automl.leaderboard(detailed=True, ensemble_only=False).reset_index()
            for rank, (_, row) in enumerate(leaderboard.head(5).iterrows(), start=1):
                records.append({
                    "Dataset": dataset,
                    "Outer fold": fold,
                    "Rank": rank,
                    "Model ID": str(row.get("model_id", row.get("index", "not reported"))),
                    "Validation cost": float(row["cost"]),
                    "Validation F1-score": 1.0 - float(row["cost"]),
                })
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)
    return records


def make_baseline_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    numerical = [col for col in X.columns if col not in categorical]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    return ColumnTransformer([("numeric", numeric, numerical), ("categorical", categorical_pipe, categorical)], sparse_threshold=0)


def evaluate_baseline(estimator, X, y):
    scores = []
    for train_idx, test_idx in outer_splits(X, y):
        pipeline = Pipeline([("preprocess", make_baseline_preprocessor(X)), ("model", estimator)])
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        probability = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        prediction = pipeline.predict(X.iloc[test_idx])
        scores.append(score_predictions(y.iloc[test_idx], prediction, probability))
    return scores


def save_table(frame, name):
    csv_path = OUT / f"{name}.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(10, len(frame.columns) * 1.8), max(2.5, 0.5 * (len(frame) + 2))))
        ax.axis("off")
        table = ax.table(cellText=frame.values, colLabels=frame.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#24476B")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F2F6FA")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    print(frame.to_string(index=False))
    print(f"Saved: {csv_path}")


def metric_cell(scores, metric):
    summary = summarize_scores(scores)
    return f"{summary[metric + ' mean']:.3f} ± {summary[metric + ' SD']:.3f}"

