import time

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .data import load_clean
from .models import get_models
from .pipeline import build_pipeline

SCORING = {"precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}


def _scores(pipe, X):
    """Score continu pour l'AUC : probabilite si disponible, sinon decision_function."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def run(use_extra=True, max_features=3000, ngram_range=(1, 1), n_splits=5) -> pd.DataFrame:
    df = load_clean()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=TEST_SIZE,
        stratify=df["target"], random_state=RANDOM_STATE,
    )
    print(f"Train : {len(X_train)} | Test : {len(X_test)} "
          f"| spam train {y_train.mean():.3%} / test {y_test.mean():.3%}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, clf in get_models().items():
        t0 = time.time()
        pipe = build_pipeline(clf, use_extra, max_features, ngram_range)

        # 1) Validation croisee sur le TRAIN uniquement
        cvr = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)

        # 2) Entrainement final sur le train, evaluation unique sur le TEST
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        rows.append({
            "Modele": name,
            "CV_F1": cvr["test_f1"].mean(), "CV_F1_std": cvr["test_f1"].std(),
            "CV_precision": cvr["test_precision"].mean(),
            "CV_recall": cvr["test_recall"].mean(),
            "CV_AUC": cvr["test_roc_auc"].mean(),
            "Test_precision": precision_score(y_test, pred, zero_division=0),
            "Test_recall": recall_score(y_test, pred),
            "Test_F1": f1_score(y_test, pred),
            "Test_AUC": roc_auc_score(y_test, _scores(pipe, X_test)),
            "FP": fp, "FN": fn,
            "temps_s": round(time.time() - t0, 1),
        })
        print(f"  {name:<20} CV_F1={rows[-1]['CV_F1']:.4f}  Test_F1={rows[-1]['Test_F1']:.4f}  "
              f"FP={fp} FN={fn}  ({rows[-1]['temps_s']}s)")

    res = pd.DataFrame(rows).sort_values("CV_F1", ascending=False).reset_index(drop=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    res.to_csv(REPORTS_DIR / "benchmark.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30)
    out = run()
    print("\n=== Classement (trie par CV_F1) ===")
    print(out.round(4).to_string(index=False))
