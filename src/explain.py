"""Explication d'une prediction : contribution de chaque indice au score (modele lineaire)."""
import numpy as np
import pandas as pd

EXTRA_NAMES = ["longueur", "part_chiffres", "part_majuscules", "symboles_monetaires", "lien"]


def feature_names(pipe) -> np.ndarray:
    tfidf = pipe.named_steps["features"].transformer_list[0][1]
    return np.array(list(tfidf.get_feature_names_out()) + EXTRA_NAMES)


def explain(pipe, text: str, k: int = 8) -> pd.DataFrame:
    """k indices les plus influents pour CE message. contribution > 0 : pousse vers spam."""
    x = pipe.named_steps["features"].transform([text])
    x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    contrib = x * pipe.named_steps["clf"].coef_.ravel()
    idx = np.nonzero(contrib)[0]
    df = pd.DataFrame({"indice": feature_names(pipe)[idx], "contribution": contrib[idx]})
    order = df["contribution"].abs().sort_values(ascending=False).index
    return df.loc[order].head(k).reset_index(drop=True)
