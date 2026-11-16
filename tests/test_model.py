import json

import joblib
import pytest

from src.config import MODELS_DIR, REPORTS_DIR
from src.explain import explain

MODEL_PATH = MODELS_DIR / "spam_model.joblib"
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="lancer d'abord python -m src.train_final")

SPAM_SMS = "WINNER!! As a valued network customer you have been selected to receive a \u00a3900 prize reward! To claim call 09061701461."
HAM_SMS = "Ok see you at 6 tonight"
HAM_EMAIL_LONG = (
    "Subject: Project update and next steps\n\nHi team,\n\nI wanted to give everyone a quick update on where we stand "
    "with the project. The first phase is finished and the results look encouraging. Anna reviewed the data on Monday "
    "and found only a couple of small inconsistencies, which she has already fixed. Next week we will start the second "
    "phase, so please send me your availability for a planning meeting by Friday. I would like to go through the "
    "schedule together, agree on who is responsible for what, and make sure nobody is overloaded during the busy "
    "period at the end of the month. If you have any questions or concerns in the meantime, feel free to call me or "
    "drop by my office. Thanks again for all your hard work, it is really appreciated.\n\nBest regards,\nTom")
PHISHING = (
    "Subject: Your account has been suspended\n\nDear customer,\n\nWe detected unusual activity on your bank account. "
    "For your security your account has been temporarily suspended. To restore access you must verify your identity "
    "within 24 hours by clicking the link below and entering your card number and password. Failure to do so will "
    "result in permanent closure of your account.\n\nhttp://secure-bank-verify.com/login\n\nSecurity Department")


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


def test_predict_proba_forme(model):
    p = model.predict_proba([SPAM_SMS, HAM_SMS])
    assert p.shape == (2, 2)
    assert ((p >= 0) & (p <= 1)).all()


def test_cas_evidents(model):
    assert model.predict_proba([SPAM_SMS])[0, 1] > 0.9
    assert model.predict_proba([HAM_SMS])[0, 1] < 0.1
    assert model.predict_proba([PHISHING])[0, 1] > 0.5


def test_non_regression_email_legitime_long_pas_classe_spam(model):
    # Avant le plafonnement de la longueur, cet email etait classe spam (~54-61 %)
    assert model.predict_proba([HAM_EMAIL_LONG])[0, 1] < 0.5


def test_explain(model):
    exp = explain(model, SPAM_SMS, k=5)
    assert list(exp.columns) == ["indice", "contribution"]
    assert 0 < len(exp) <= 5


def test_metriques_minimales_du_dernier_entrainement():
    m = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert m["f1"] >= 0.93 and m["recall"] >= 0.88 and m["precision"] >= 0.95 and m["roc_auc"] >= 0.99
