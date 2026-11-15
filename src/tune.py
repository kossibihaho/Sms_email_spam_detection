"""Etape 4b : reglage des hyperparametres (GridSearchCV sur le TRAIN) des meilleures familles."""
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

from .config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .data import load_clean
from .pipeline import build_pipeline

MF = "features__tfidf__max_features"

CANDIDATS = {
    "LinearSVC": (LinearSVC(random_state=RANDOM_STATE), {
        "clf__C": [0.1, 0.3, 1, 3, 10],
        "clf__class_weight": [None, "balanced"],
        MF: [1000, 3000, None],
    }),
    "LogisticRegression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), {
        "clf__C": [1, 3, 10, 30, 100],
        "clf__class_weight": [None, "balanced"],
        MF: [1000, 3000, None],
    }),
    "BernoulliNB": (BernoulliNB(), {
        "clf__alpha": [0.1, 0.3, 1.0],
        MF: [500, 1000, 2000, 3000],
    }),
}


def _scores(pipe, X):
    return pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load_clean()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=TEST_SIZE, stratify=df["target"], random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    rows, best_estimators = [], {}
    for name, (clf, grid) in CANDIDATS.items():
        gs = GridSearchCV(build_pipeline(clf), grid, scoring="f1", cv=cv, refit=True, n_jobs=1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        pred = best.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({
            "Modele": name,
            "CV_F1": gs.best_score_,
            "CV_F1_std": gs.cv_results_["std_test_score"][gs.best_index_],
            "Test_precision": precision_score(y_test, pred),
            "Test_recall": recall_score(y_test, pred),
            "Test_F1": f1_score(y_test, pred),
            "Test_AUC": roc_auc_score(y_test, _scores(best, X_test)),
            "FP": fp, "FN": fn,
            "meilleurs_params": json.dumps(gs.best_params_, default=str),
        })
        best_estimators[name] = best
        print(f"{name:<20} CV_F1={gs.best_score_:.4f}  Test_F1={rows[-1]['Test_F1']:.4f}  "
              f"FP={fp} FN={fn}  params={gs.best_params_}")

    res = pd.DataFrame(rows).sort_values("CV_F1", ascending=False).reset_index(drop=True)
    res.to_csv(REPORTS_DIR / "tuning.csv", index=False)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_estimators, MODELS_DIR / "_candidates.joblib")
    print("\n=== Candidats regles (tries par CV_F1) ===")
    print(res.drop(columns="meilleurs_params").round(4).to_string(index=False))
    print("\nGagnant selon la CV :", res.loc[0, "Modele"])
