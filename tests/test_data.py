from src.data import load_clean, load_raw


def test_dimensions_apres_nettoyage():
    assert load_raw().shape == (5572, 2)
    df = load_clean()
    assert df.shape == (5169, 3)
    assert list(df.columns) == ["label", "text", "target"]


def test_repartition_des_classes():
    df = load_clean()
    assert df["target"].sum() == 653           # spams
    assert (df["target"] == 0).sum() == 4516   # hams


def test_pas_de_doublons_ni_de_na_et_artefact_corrige():
    df = load_clean()
    assert not df.duplicated().any()
    assert not df.isna().any().any()
    assert not df["text"].str.contains("\u00e5\u00a3").any()   # "a-rond + livre" doit etre devenu "livre"
