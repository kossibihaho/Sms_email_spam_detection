import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler


def extra_features(texts) -> np.ndarray:
    """Signaux 'non lexicaux' typiques du spam. Toutes les valeurs sont >= 0."""
    rows = []
    for t in texts:
        t = str(t)
        n = len(t) or 1
        low = t.lower()
        rows.append([
            np.log1p(len(t)),                                 # longueur
            sum(c.isdigit() for c in t) / n,                  # part de chiffres
            sum(c.isupper() for c in t) / n,                  # part de majuscules
            sum(t.count(s) for s in "\u00a3$\u20ac"),         # symboles monetaires
            float("http" in low or "www" in low),             # presence d'un lien
        ])
    return np.array(rows, dtype=float)


def extra_block() -> Pipeline:
    return Pipeline([
        ("fn", FunctionTransformer(extra_features)),
        ("scale", MaxAbsScaler()),   # garde les valeurs >= 0 (necessaire pour les Naive Bayes)
    ])
