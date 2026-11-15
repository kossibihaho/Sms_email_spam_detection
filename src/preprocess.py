import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    _STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOP = set(stopwords.words("english"))

_stemmer = PorterStemmer()
_TOKEN = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def clean_text(text: str) -> str:
    """minuscules -> tokens alphanumeriques -> sans stopwords -> stemming."""
    tokens = _TOKEN.findall(str(text).lower())
    return " ".join(_stem(t) for t in tokens if t not in _STOP)
