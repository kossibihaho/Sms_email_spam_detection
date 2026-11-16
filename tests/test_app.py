import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODELS_DIR, ROOT

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "spam_model.joblib").exists(),
                                reason="lancer d'abord python -m src.train_final")
APP = str(ROOT / "app" / "streamlit_app.py")


def test_toutes_les_pages_se_chargent():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    for page in ["Analyse par lots", "Performances", "\u00c0 propos", "D\u00e9tecter"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_detection_spam_et_ham():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="texte").set_value("WINNER!! You won a \u00a3900 prize, call 09061701461 now").run()
    [b for b in at.button if b.label == "Analyser"][0].click().run()
    assert at.error and not at.exception
    at.text_area(key="texte").set_value("Ok see you at 6 tonight").run()
    [b for b in at.button if b.label == "Analyser"][0].click().run()
    assert at.success and not at.exception
