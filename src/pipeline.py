from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from .features import extra_block
from .preprocess import clean_text


def build_pipeline(clf, use_extra=True, max_features=3000, ngram_range=(1, 1)) -> Pipeline:
    """Texte brut -> (TF-IDF [+ features extra]) -> classifieur.
    Tout est ajuste uniquement sur les donnees d'entrainement."""
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    feats = FeatureUnion([("tfidf", tfidf), ("extra", extra_block())]) if use_extra else tfidf
    return Pipeline([("features", feats), ("clf", clf)])
