import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # le modèle a besoin de spam_utils.py, situé à la racine du projet
from spam_utils import EXTRA_NAMES     # noqa: E402

st.set_page_config(page_title="Détection de spam")

MODEL_PATH = ROOT / "models" / "spam_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

EXEMPLES = {
    "Exemple de spam 1": "WINNER!! As a valued network customer you have been selected to receive a \u00a3900 prize reward! To claim call 09061701461.",
    "Exemple de spam 2": "FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry question(std txt rate)",
    "Exemple de message normal 1": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "Exemple de message normal 2": "Ok see you at 6 tonight",
}

NOMS_CARACTERISTIQUES = {
    "longueur": "longueur du message",
    "part_chiffres": "part de chiffres",
    "part_majuscules": "part de majuscules",
    "symboles_monetaires": "symboles monétaires (£, $, €)",
    "lien": "présence d'un lien",
}

MOTS_FRANCAIS = {"le", "la", "les", "des", "est", "pour", "vous", "nous", "avec", "une", "dans", "pas",
                 "bonjour", "merci", "votre", "sur", "que", "qui", "je", "tu", "mon", "ton"}


def pct(valeur):
    return f"{valeur * 100:.1f} %".replace(".", ",")


@st.cache_resource
def charger_modele():
    return joblib.load(MODEL_PATH)


def semble_francais(texte):
    mots = {m.strip(".,!?;:'\"").lower() for m in texte.split()}
    return len(mots & MOTS_FRANCAIS) >= 2


def expliquer(modele, texte, k=8):
    """Éléments du message qui ont le plus pesé dans le score (poids = valeur x coefficient)."""
    x = modele.named_steps["features"].transform([texte])
    x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    poids = x * modele.named_steps["clf"].coef_.ravel()
    tfidf = modele.named_steps["features"].transformer_list[0][1]
    noms = list(tfidf.get_feature_names_out()) + EXTRA_NAMES
    lignes = [(NOMS_CARACTERISTIQUES.get(noms[i], noms[i]),
               "spam" if poids[i] > 0 else "message normal",
               round(float(poids[i]), 2))
              for i in np.nonzero(poids)[0]]
    lignes.sort(key=lambda l: abs(l[2]), reverse=True)
    return pd.DataFrame(lignes[:k], columns=["Élément", "Pousse vers", "Poids"])


def page_analyse(modele, seuil):
    st.header("Analyser un message")
    st.write("Collez un SMS ou un email écrit en anglais, puis cliquez sur « Analyser ».")

    colonnes = st.columns(len(EXEMPLES))
    for colonne, (nom, texte) in zip(colonnes, EXEMPLES.items()):
        if colonne.button(nom):
            st.session_state["texte"] = texte

    texte = st.text_area("Message", key="texte", height=150)
    if not st.button("Analyser", type="primary"):
        return
    if not texte.strip():
        st.warning("Le message est vide.")
        return

    score = float(modele.predict_proba([texte])[0, 1])
    if score >= seuil:
        st.error(f"Ce message est classé comme spam. Score de spam : {pct(score)} (seuil : {pct(seuil)}).")
    else:
        st.success(f"Ce message est classé comme normal. Score de spam : {pct(score)} (seuil : {pct(seuil)}).")
    if semble_francais(texte):
        st.warning("Ce message semble écrit en français. Le modèle a été entraîné sur des messages en anglais : "
                   "le résultat n'est pas fiable.")

    st.subheader("Ce qui a le plus pesé dans le score")
    explication = expliquer(modele, texte)
    if explication.empty:
        st.write("Aucun élément connu du modèle dans ce message.")
    else:
        st.dataframe(explication, hide_index=True)
        st.caption("Les mots sont affichés après réduction à leur racine (par exemple « mobil » pour « mobile »). "
                   "Un poids positif pousse vers le spam, un poids négatif vers le message normal.")


def page_lot(modele, seuil):
    st.header("Analyser plusieurs messages")
    onglet_texte, onglet_csv = st.tabs(["Coller des messages", "Importer un fichier CSV"])
    textes = None

    with onglet_texte:
        brut = st.text_area("Un message par ligne", height=200, key="lot")
        if st.button("Analyser ces messages") and brut.strip():
            textes = [ligne for ligne in brut.splitlines() if ligne.strip()]

    with onglet_csv:
        fichier = st.file_uploader("Fichier CSV", type="csv")
        if fichier is not None:
            try:
                donnees = pd.read_csv(fichier)
            except UnicodeDecodeError:
                fichier.seek(0)
                donnees = pd.read_csv(fichier, encoding="cp1252")
            colonne = st.selectbox("Colonne qui contient les messages", donnees.columns)
            if st.button("Analyser le fichier"):
                textes = donnees[colonne].dropna().astype(str).tolist()

    if textes:
        scores = modele.predict_proba(textes)[:, 1]
        resultat = pd.DataFrame({
            "Message": textes,
            "Score de spam": scores.round(3),
            "Classement": ["spam" if s >= seuil else "normal" for s in scores],
        })
        nb_spam = int((resultat["Classement"] == "spam").sum())
        a, b, c = st.columns(3)
        a.metric("Messages analysés", len(resultat))
        b.metric("Classés spam", nb_spam)
        c.metric("Part de spam", pct(nb_spam / len(resultat)))
        st.dataframe(resultat, hide_index=True)
        st.download_button("Télécharger les résultats (CSV)", resultat.to_csv(index=False).encode("utf-8"),
                           "resultats.csv", "text/csv")


def page_performances():
    st.header("Performances du modèle")
    if not METRICS_PATH.exists():
        st.warning("Fichier reports/metrics.json introuvable. Exécutez d'abord le notebook.")
        return
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.write(f"Modèle : {m['modele']}. Évaluation sur {m['n_test']} messages que le modèle n'a pas vus "
             "pendant son entraînement.")

    colonnes = st.columns(5)
    for colonne, (nom, cle) in zip(colonnes, [("Précision", "precision"), ("Rappel", "recall"), ("F1", "f1"),
                                              ("AUC ROC", "roc_auc"), ("AUC PR", "pr_auc")]):
        colonne.metric(nom, f"{m[cle]:.3f}".replace(".", ","))

    mc = m["matrice_confusion"]
    st.subheader("Matrice de confusion")
    st.dataframe(pd.DataFrame([[mc["TN"], mc["FP"]], [mc["FN"], mc["TP"]]],
                              index=["Vrai message normal", "Vrai spam"],
                              columns=["Classé normal", "Classé spam"]))
    st.write(f"{mc['FP']} message(s) normal(aux) classé(s) à tort comme spam, "
             f"{mc['FN']} spam(s) non détecté(s).")

    for titre, fichier in [("Comparaison des 14 modèles (validation croisée)", "benchmark.csv"),
                           ("Modèles réglés et ensembles", "reglage.csv"),
                           ("Expériences (équilibrage, caractéristiques, vocabulaire)", "experiences.csv")]:
        chemin = ROOT / "reports" / fichier
        if chemin.exists():
            with st.expander(titre):
                st.dataframe(pd.read_csv(chemin).round(3), hide_index=True)


def page_a_propos():
    st.header("À propos")
    st.markdown("""
Projet de Machine Learning : détection de SMS et d'emails indésirables (spam).

**Réalisé par**
- AGBEVO Kodjo Honoré ([@Honore-M12](https://github.com/Honore-M12))
- BIHAHO Kossi ([@kossibihaho](https://github.com/kossibihaho))
- IRERA Furaha Jospine ([@jospine463](https://github.com/jospine463/))
- RAKOTOMANGA Manohisoa Mireille ([@soamanohy](https://github.com/soamanohy))

**Encadrant** : Prof. Mr [Ayoub Charef](https://www.linkedin.com/in/ayoub-charef-531897128/)

**Données** : SMS Spam Collection, 5 572 SMS en anglais (5 169 après suppression des doublons), dont 12,6 % de spams.

**Méthode** : nettoyage du texte, TF-IDF et cinq caractéristiques simples (longueur, chiffres, majuscules,
symboles monétaires, lien), régression logistique. Les spams d'entraînement sont complétés par des spams générés
(SMOTE) pour équilibrer les deux classes.

**Limites**
- Le modèle n'a vu que des SMS en anglais : les messages en français ne sont pas fiables.
- Il n'a jamais été évalué sur de vrais emails.
- Le jeu de données est ancien : il contient peu de spams récents (hameçonnage, faux colis, cryptomonnaies).
""")


def main():
    st.sidebar.title("Détection de spam")
    page = st.sidebar.radio("Navigation", ["Analyser un message", "Analyser plusieurs messages",
                                           "Performances du modèle", "À propos"])
    seuil = st.sidebar.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05,
                              help="Un message est classé spam si son score dépasse ce seuil. "
                                   "Plus le seuil est haut, moins il y a de faux positifs, mais plus de spams passent.")
    if not MODEL_PATH.exists():
        st.error("Modèle introuvable (models/spam_model.joblib). Exécutez d'abord le notebook en entier.")
        st.stop()
    modele = charger_modele()

    if page == "Analyser un message":
        page_analyse(modele, seuil)
    elif page == "Analyser plusieurs messages":
        page_lot(modele, seuil)
    elif page == "Performances du modèle":
        page_performances()
    else:
        page_a_propos()


main()
