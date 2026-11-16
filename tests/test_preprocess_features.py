import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import MAX_LEN, extra_features
from src.pipeline import build_pipeline
from src.preprocess import clean_text


def test_clean_text_minuscules_stopwords_stemming():
    assert clean_text("I'm gonna be HOME soon!!!") == "gonna home soon"
    assert clean_text("Winning WINNERS") == "win winner"


def test_clean_text_cas_limites():
    assert clean_text("") == ""
    assert clean_text("!!! ??? ...") == ""
    assert clean_text("the and of") == ""          # que des stopwords


def test_features_extra_forme_et_positivite():
    X = extra_features(["Hello", "WIN \u00a3500 now www.x.com 12345", ""])
    assert X.shape == (3, 5)
    assert (X >= 0).all()                           # exige par les Naive Bayes
    assert X[1, 3] == 1 and X[1, 4] == 1            # symbole monetaire et lien detectes


def test_longueur_plafonnee():
    court, moyen, long_, tres_long = (extra_features(["a" * n])[0, 0] for n in (10, 100, MAX_LEN + 40, 900))
    assert court < moyen                            # croissante sous le plafond
    assert long_ == tres_long                       # saturee au-dela (corrige les faux positifs sur emails longs)


def test_pas_de_fuite_de_donnees_le_vocabulaire_vient_du_train():
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(pd.Series(["free prize win", "see you tonight", "call now free", "lunch tomorrow ok"]), [1, 0, 1, 0])
    vocab = pipe.named_steps["features"].transformer_list[0][1].vocabulary_
    pipe.predict(["zebra zebra zebra"])             # mot inconnu : pas d'erreur
    assert "zebra" not in vocab                      # et predict n'apprend rien
