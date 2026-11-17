"""Fonctions de pretraitement partagees entre le notebook et l'application Streamlit.
Fichier ecrit par le notebook : le modele sauvegarde reference ces fonctions par leur nom."""
import re
from functools import lru_cache

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    _STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOP = set(stopwords.words("english"))

_stemmer = PorterStemmer()
_TOKEN = re.compile(r"[a-z0-9]+")

# Plafond de longueur (caracteres) : 95 % des hams font moins de 160 caracteres et
# la mediane des spams est de 148. Au-dela, la longueur n'informe plus.
MAX_LEN = 160
EXTRA_NAMES = ["longueur", "part_chiffres", "part_majuscules", "symboles_monetaires", "lien"]


@lru_cache(maxsize=None)
def _stem(word):
    return _stemmer.stem(word)


def clean_text(text):
    """minuscules -> tokens alphanumeriques -> sans stopwords -> stemming."""
    tokens = _TOKEN.findall(str(text).lower())
    return " ".join(_stem(t) for t in tokens if t not in _STOP)


def extra_features(texts):
    """5 signaux non lexicaux typiques du spam. Toutes les valeurs sont >= 0."""
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
