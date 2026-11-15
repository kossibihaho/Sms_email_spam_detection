"""Etape 4c : modele final, metriques, figures, sauvegarde des artefacts."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")  # pas d'affichage : on exporte seulement les images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .data import load_clean

MODEL = "LogisticRegression"   # choix documente dans le README (probabilites natives + interpretable)
EXTRA_NAMES = ["longueur", "part_chiffres", "part_majuscules", "symboles_monetaires", "lien"]


def feature_names(pipe):
    tfidf = pipe.named_steps["features"].transformer_list[0][1]
    return np.array(list(tfidf.get_feature_names_out()) + EXTRA_NAMES)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_clean()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=TEST_SIZE, stratify=df["target"], random_state=RANDOM_STATE)

    # 1) Modele regle a l'etape 4b, entraine sur le TRAIN -> metriques honnetes sur le TEST
    best = joblib.load(MODELS_DIR / "_candidates.joblib")[MODEL]
    assert hasattr(best, "predict_proba"), f"{MODEL} n'a pas de predict_proba"
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "modele": MODEL,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "matrice_confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "parametres": {k: str(v) for k, v in best.get_params().items()
                       if k in ("clf__C", "clf__class_weight", "features__tfidf__max_features")},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 2) Figures
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["ham", "spam"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matrice de confusion (test) - {MODEL}")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title=f"ROC (AUC = {metrics['roc_auc']})", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs")
    axes[1].plot(rec, prec)
    axes[1].set(title=f"Precision-Rappel (AP = {metrics['pr_auc']})", xlabel="Rappel", ylabel="Precision")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "roc_pr.png", dpi=150); plt.close(fig)

    bench = pd.read_csv(REPORTS_DIR / "benchmark.csv").sort_values("CV_F1")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(bench["Modele"], bench["CV_F1"], xerr=bench["CV_F1_std"], color="steelblue")
    ax.set(xlim=(0.6, 1.0), xlabel="F1 en validation croisee (moyenne +/- ecart-type)",
           title="Comparaison des 14 modeles (avant reglage)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "benchmark_f1.png", dpi=150); plt.close(fig)

    names, coefs = feature_names(best), best.named_steps["clf"].coef_.ravel()
    order = np.argsort(coefs)
    top_spam, top_ham = order[-15:], order[:15]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(names[top_spam], coefs[top_spam], color="crimson"); axes[0].set_title("Indices de SPAM")
    axes[1].barh(names[top_ham], coefs[top_ham], color="seagreen"); axes[1].set_title("Indices de HAM")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "top_words.png", dpi=150); plt.close(fig)

    # 3) Modele de deploiement : meme configuration, reentraine sur TOUTES les donnees
    final = clone(best).fit(df["text"], df["target"])
    joblib.dump(final, MODELS_DIR / "spam_model.joblib")
    print("\nSauvegarde :", MODELS_DIR / "spam_model.joblib")
    print("Test rapide :", final.predict_proba(["WINNER!! Claim your \u00a31000 prize now http://win.co",
                                                "Ok see you at 6 tonight"])[:, 1].round(3))


if __name__ == "__main__":
    main()
