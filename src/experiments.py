"""Etape 4a : experiences controlees (une variable a la fois) sur le TRAIN, en validation croisee."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .data import load_clean
from .pipeline import build_pipeline

SCORING = {"precision": "precision", "recall": "recall", "f1": "f1"}

# (nom de l'experience, parametres du pipeline). La premiere ligne est la reference.
EXPERIENCES = [
    ("reference (extra, 3000, uni)",   dict(use_extra=True,  max_features=3000, ngram_range=(1, 1))),
    ("sans features extra",            dict(use_extra=False, max_features=3000, ngram_range=(1, 1))),
    ("max_features=1000",              dict(use_extra=True,  max_features=1000, ngram_range=(1, 1))),
    ("max_features=5000",              dict(use_extra=True,  max_features=5000, ngram_range=(1, 1))),
    ("max_features=None",              dict(use_extra=True,  max_features=None, ngram_range=(1, 1))),
    ("ngram (1,2)",                    dict(use_extra=True,  max_features=3000, ngram_range=(1, 2))),
    ("ngram (1,2) + max_feat 5000",    dict(use_extra=True,  max_features=5000, ngram_range=(1, 2))),
]

MODELES = {
    "LinearSVC":          lambda: LinearSVC(random_state=RANDOM_STATE),
    "LinearSVC_balanced": lambda: LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    "LogReg_balanced":    lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "BernoulliNB":        lambda: BernoulliNB(),
}

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load_clean()
    X_train, _, y_train, _ = train_test_split(
        df["text"], df["target"], test_size=TEST_SIZE, stratify=df["target"], random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for mname, make in MODELES.items():
        for ename, params in EXPERIENCES:
            r = cross_validate(build_pipeline(make(), **params), X_train, y_train, cv=cv, scoring=SCORING)
            rows.append({"Modele": mname, "Experience": ename,
                         "F1": r["test_f1"].mean(), "F1_std": r["test_f1"].std(),
                         "Precision": r["test_precision"].mean(), "Recall": r["test_recall"].mean()})
        print(f"{mname} ok")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "experiments.csv", index=False)
    for m in MODELES:
        print(f"\n=== {m} ===")
        print(res[res.Modele == m].drop(columns="Modele").round(4).to_string(index=False))
