import pandas as pd
from .config import DATA_PATH


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut : 2 colonnes utiles, encodage cp1252."""
    df = pd.read_csv(path, encoding="cp1252", usecols=[0, 1])
    df.columns = ["label", "text"]
    return df


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    """Charge, repare le symbole livre, supprime NA et doublons, ajoute la cible."""
    df = load_raw(path)
    # Le fichier source contient "\u00e5\u00a3" a la place de "\u00a3" (defaut du dataset d'origine)
    df["text"] = df["text"].str.replace("\u00e5\u00a3", "\u00a3", regex=False)
    df = df.dropna().drop_duplicates(keep="first").reset_index(drop=True)
    df["target"] = (df["label"] == "spam").astype(int)
    return df


if __name__ == "__main__":
    raw = load_raw()
    clean = load_clean()
    print("Brut             :", raw.shape)
    print("Nettoye          :", clean.shape, "(label, text, target)")
    print("Doublons retires :", len(raw) - len(clean))
    print("\nRepartition :")
    print(clean["label"].value_counts())
    print(clean["label"].value_counts(normalize=True).round(4))
    print("\nMessages avec l'artefact (devrait etre 0) :",
          clean["text"].str.contains("\u00e5\u00a3").sum())
    print("Messages avec le symbole livre            :",
          clean["text"].str.contains("\u00a3").sum())
