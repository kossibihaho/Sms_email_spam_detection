import json
import sys
from pathlib import Path

# Le modèle sauvegardé référence src.preprocess / src.features : on ajoute la racine du dépôt au path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explain import explain

st.set_page_config(page_title="Détecteur de spam SMS", page_icon="📩", layout="wide")

MODEL_PATH = ROOT / "models" / "spam_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIG = ROOT / "reports" / "figures"

EXEMPLES = {
    "Spam : gain": "WINNER!! As a valued network customer you have been selected to receive a \u00a3900 prize reward! To claim call 09061701461.",
    "Spam : offre": "FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry question(std txt rate)",
    "Ham : rendez-vous": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "Ham : court": "Ok see you at 6 tonight",
}

MOTS_FR = {"le", "la", "les", "des", "est", "pour", "vous", "nous", "avec", "une", "dans", "pas",
           "bonjour", "merci", "votre", "sur", "que", "qui", "je", "tu", "mon", "ton"}


@st.cache_resource
def charger_modele():
    return joblib.load(MODEL_PATH)


def semble_francais(texte: str) -> bool:
    mots = {m.strip(".,!?;:'\"").lower() for m in texte.split()}
    return len(mots & MOTS_FR) >= 2


def jauge(score: float, seuil: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100, number={"suffix": " %"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "crimson" if score >= seuil else "seagreen"},
               "threshold": {"line": {"color": "black", "width": 3}, "value": seuil * 100}},
        title={"text": "Score de spam"}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ---------------------------------------------------------------- pages
def page_detecter(model, seuil):
    st.header("Analyser un message")
    cols = st.columns(len(EXEMPLES))
    for col, (nom, txt) in zip(cols, EXEMPLES.items()):
        if col.button(nom, width="stretch"):
            st.session_state["texte"] = txt

    texte = st.text_area("Message (SMS ou email, en anglais)", key="texte", height=130,
                         placeholder="Collez ou saisissez un message...")
    if st.button("Analyser", type="primary"):
        if not texte.strip():
            st.warning("Saisissez un message.")
            return
        score = float(model.predict_proba([texte])[:, 1][0])
        est_spam = score >= seuil

        g, d = st.columns([1, 1])
        with g:
            st.plotly_chart(jauge(score, seuil), width="stretch")
        with d:
            (st.error if est_spam else st.success)(
                f"**{'SPAM' if est_spam else 'HAM (message légitime)'}**  \nScore : {score:.1%}  |  Seuil : {seuil:.0%}")
            st.caption("Le modèle est entraîné avec `class_weight='balanced'` : "
                       "c'est un score de classement, pas une probabilité calibrée.")
            if semble_francais(texte):
                st.warning("Ce message semble écrit en français. Le modèle a été entraîné sur des SMS "
                           "en **anglais** : le résultat n'est pas fiable.")

        st.subheader("Pourquoi ce score ?")
        exp = explain(model, texte)
        if exp.empty:
            st.info("Aucun indice connu du modèle dans ce message.")
        else:
            exp["effet"] = exp["contribution"].apply(lambda v: "pousse vers SPAM" if v > 0 else "pousse vers HAM")
            fig = px.bar(exp.iloc[::-1], x="contribution", y="indice", color="effet", orientation="h",
                         color_discrete_map={"pousse vers SPAM": "crimson", "pousse vers HAM": "seagreen"})
            fig.update_layout(height=350, yaxis_title=None, legend_title=None)
            st.plotly_chart(fig, width="stretch")
            st.caption("Les mots sont affichés après stemming (ex. *mobil* pour *mobile*). "
                       "`longueur`, `part_chiffres`... sont des caractéristiques calculées sur le message.")


def predire_lot(model, textes, seuil):
    scores = model.predict_proba(textes)[:, 1]
    return pd.DataFrame({"message": textes, "score_spam": scores.round(4),
                         "prediction": ["spam" if s >= seuil else "ham" for s in scores]})


def page_lot(model, seuil):
    st.header("Analyse par lots")
    tab1, tab2 = st.tabs(["Coller des messages", "Importer un CSV"])
    textes = None
    with tab1:
        brut = st.text_area("Un message par ligne", height=180, key="lot_txt")
        if st.button("Analyser les lignes", key="b1") and brut.strip():
            textes = [l for l in brut.splitlines() if l.strip()]
    with tab2:
        f = st.file_uploader("Fichier CSV", type="csv")
        if f is not None:
            try:
                df = pd.read_csv(f)
            except UnicodeDecodeError:
                f.seek(0)
                df = pd.read_csv(f, encoding="cp1252")
            col = st.selectbox("Colonne contenant le texte", df.columns)
            if st.button("Analyser le CSV", key="b2"):
                textes = df[col].dropna().astype(str).tolist()
    if textes:
        res = predire_lot(model, textes, seuil)
        n_spam = int((res["prediction"] == "spam").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages", len(res)); c2.metric("Spams détectés", n_spam)
        c3.metric("Part de spam", f"{n_spam / len(res):.1%}")
        st.dataframe(res, width="stretch", hide_index=True)
        st.download_button("Télécharger les résultats (CSV)", res.to_csv(index=False).encode("utf-8"),
                           "resultats_spam.csv", "text/csv")


def page_performances():
    st.header("Performances du modèle")
    if not METRICS_PATH.exists():
        st.warning("Lancez d'abord `python -m src.train_final`.")
        return
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.caption(f"Modèle : **{m['modele']}** | évaluation sur {m['n_test']} messages jamais vus à l'entraînement")
    c = st.columns(5)
    for col, (lab, key) in zip(c, [("Précision", "precision"), ("Rappel", "recall"), ("F1", "f1"),
                                    ("AUC ROC", "roc_auc"), ("AUC PR", "pr_auc")]):
        col.metric(lab, f"{m[key]:.3f}")
    mc = m["matrice_confusion"]
    st.write(f"**{mc['FP']}** message(s) légitime(s) classé(s) spam (faux positifs) et "
             f"**{mc['FN']}** spam(s) non détecté(s) (faux négatifs) sur {m['n_test']}.")
    a, b = st.columns(2)
    if (FIG / "confusion_matrix.png").exists():
        a.image(str(FIG / "confusion_matrix.png"))
    if (FIG / "top_words.png").exists():
        b.image(str(FIG / "top_words.png"))
    for nom in ("roc_pr.png", "benchmark_f1.png"):
        if (FIG / nom).exists():
            st.image(str(FIG / nom))
    for titre, f in [("Benchmark des 14 modèles (CV 5 plis)", "benchmark.csv"),
                     ("Candidats réglés", "tuning.csv"), ("Expériences", "experiments.csv")]:
        p = ROOT / "reports" / f
        if p.exists():
            with st.expander(titre):
                st.dataframe(pd.read_csv(p), width="stretch", hide_index=True)


def page_apropos():
    st.header("À propos")
    st.markdown("""
**Projet de Machine Learning** : détection de SMS/emails indésirables (spam).

**Réalisé par**
- AGBEVO Kodjo Honoré ([@Honore-M12](https://github.com/Honore-M12))
- BIHAHO Kossi ([@kossibihaho](https://github.com/kossibihaho))
- IRERA Furaha Jospine ([@jospine463](https://github.com/jospine463/))
- RAKOTOMANGA Manohisoa Mireille ([@soamanohy](https://github.com/soamanohy))

**Encadrant** : Prof. Mr [Ayoub Charef](https://www.linkedin.com/in/ayoub-charef-531897128/)

**Données** : SMS Spam Collection, 5 572 SMS en anglais (5 169 après suppression des doublons), 12,6 % de spams.

**Méthode** : nettoyage, TF-IDF + 5 caractéristiques (longueur, chiffres, majuscules, symboles monétaires, lien),
régression logistique réglée par validation croisée.

**Limites**
- Entraînement sur des SMS **anglais** uniquement : les messages en français ne sont pas fiables.
- Le jeu de données est ancien : les spams actuels (phishing, livraisons, cryptos) y sont peu représentés.
- Le modèle a appris un artefact du jeu de données (`gt`/`lt`, restes d'entités HTML dans des SMS légitimes).
""")


# ---------------------------------------------------------------- main
def main():
    st.sidebar.title("📩 Détecteur de spam")
    page = st.sidebar.radio("Navigation", ["Détecter", "Analyse par lots", "Performances", "À propos"])
    seuil = st.sidebar.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05,
                              help="Plus le seuil est haut, moins on classe de messages en spam "
                                   "(moins de faux positifs, plus de spams manqués).")
    if not MODEL_PATH.exists():
        st.error("Modèle introuvable. Lancez : `python -m src.train_final`")
        st.stop()
    model = charger_modele()
    if page == "Détecter":
        page_detecter(model, seuil)
    elif page == "Analyse par lots":
        page_lot(model, seuil)
    elif page == "Performances":
        page_performances()
    else:
        page_apropos()


main()
