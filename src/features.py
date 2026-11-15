import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

# Plafond de longueur (caracteres). Dans le dataset, 95 % des hams font moins de 160 caracteres et
# les spams ont une mediane de 148 : au-dela, la longueur n'apporte plus d'information utile.
# Sans plafond, un email legitime long etait classe spam (le modele apprenait "long = suspect").
MAX_LEN = 160


def extra_features(texts) -> np.ndarray:
    """Signaux 'non lexicaux' typiques du spam. Toutes les valeurs sont >= 0."""
    rows = []
    for t in texts:
        t = str(t)
        n = len(t) or 1
        low = t.lower()
        rows.append([
            np.log1p(min(len(t), MAX_LEN)),                   # longueur (plafonnee)
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
