# Détection de SMS et d'emails indésirables (spam)

Projet académique de groupe en Machine Learning : détecter et classifier les SMS et emails indésirables (spam) à l'aide de plusieurs algorithmes d'apprentissage automatique. Le travail principal est dans un notebook ; une petite application Streamlit permet d'essayer le modèle final.

## Équipe

| Nom | GitHub |
|---|---|
| AGBEVO Kodjo Honoré | [@Honore-M12](https://github.com/Honore-M12) |
| BIHAHO Kossi | [@kossibihaho](https://github.com/kossibihaho) |
| IRERA Furaha Jospine | [@jospine463](https://github.com/jospine463/) |
| RAKOTOMANGA Manohisoa Mireille | [@soamanohy](https://github.com/soamanohy) |

Encadrant : Prof. Mr [Ayoub Charef](https://www.linkedin.com/in/ayoub-charef-531897128/)

## Contenu du dépôt

```
sms-spam-detection.ipynb   notebook principal : nettoyage, analyse, modèles, résultats
spam.csv                   jeu de données (SMS en anglais)
spam_utils.py              fonctions de prétraitement, écrites par le notebook et utilisées par l'application
app.py                     application Streamlit
models/spam_model.joblib   modèle final, produit par le notebook
reports/                   résultats : tableaux (CSV), métriques (JSON) et figures (PNG)
requirements.txt           bibliothèques nécessaires
```

## Installation

Avec Git Bash :

```bash
git clone https://github.com/kossibihaho/Sms_email_spam_detection.git
cd Sms_email_spam_detection
python -m venv .venv
source .venv/Scripts/activate        # Linux et macOS : source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

1. **Notebook** : lancer `jupyter notebook`, ouvrir `sms-spam-detection.ipynb`, puis *Kernel > Restart & Run All*. L'exécution complète prend environ 6 à 7 minutes. Elle crée `spam_utils.py`, `models/` et `reports/`.
2. **Application** : `streamlit run app.py`.

Le dépôt contient déjà les fichiers produits par le notebook : l'application peut être lancée sans le réexécuter.

## Données

SMS Spam Collection : 5 572 SMS en anglais, 5 169 après suppression de 403 doublons. On y trouve 4 516 messages normaux (87,4 %) et 653 spams (12,6 %). Le fichier contient un défaut d'encodage (« å£ » à la place de « £ »), corrigé au chargement.

## Méthode

1. Nettoyage des données et analyse exploratoire (longueur des messages, mots les plus fréquents).
2. Prétraitement du texte : minuscules, mots alphanumériques, suppression des mots vides, réduction des mots à leur racine (Porter).
3. Séparation entraînement / test en premier (80 % / 20 %, mêmes proportions de spam). Tout ce qui est appris (vocabulaire, échelles, spams générés) l'est sur l'entraînement uniquement, grâce à un `Pipeline`.
4. Variables : TF-IDF et cinq caractéristiques simples (longueur plafonnée à 160 caractères, part de chiffres, part de majuscules, symboles monétaires, présence d'un lien).
5. Équilibrage des classes : SMOTE génère des spams synthétiques dans l'ensemble d'entraînement (de 522 à 3 613 spams). Le jeu de test n'est jamais modifié.
6. Comparaison de 14 modèles par validation croisée à 5 plis, réglage de trois d'entre eux, puis deux ensembles (Voting et Stacking).
7. Modèle final : régression logistique.

## Résultats

Évaluation sur 1 034 messages (dont 131 spams) que le modèle n'a pas vus pendant son entraînement :

| Précision | Rappel | F1 | AUC ROC | Messages normaux classés spam | Spams non détectés |
|---|---|---|---|---|---|
| 0,984 | 0,931 | 0,957 | 0,997 | 2 | 9 |

Ce que les expériences ont montré (régression logistique, validation croisée sur l'entraînement) :

| Équilibrage des classes | F1 | Rappel | Précision |
|---|---|---|---|
| Aucun | 0,933 | 0,889 | 0,983 |
| Poids des classes (`class_weight`) | 0,956 | 0,946 | 0,965 |
| Duplication aléatoire des spams | 0,955 | 0,944 | 0,967 |
| SMOTE (spams générés) | 0,959 | 0,943 | 0,977 |

- Équilibrer les classes améliore nettement le rappel (de 0,889 à environ 0,945). Les trois méthodes donnent des résultats équivalents : leurs écarts sont plus petits que l'écart-type (environ 0,005).
- Les cinq caractéristiques supplémentaires font passer le F1 de 0,899 à 0,959.
- La taille du vocabulaire et les n-grammes n'ont pas d'effet visible (F1 entre 0,955 et 0,959).
- Après réglage, la régression logistique (0,961), le stacking (0,961), LinearSVC (0,960) et le voting (0,957) sont à égalité compte tenu de l'écart-type. Nous avons gardé la régression logistique parce que ses coefficients permettent à l'application d'expliquer chaque décision.

Les chiffres peuvent varier légèrement selon la version des bibliothèques.

## Ce que nous avons corrigé par rapport au premier notebook

| Dans le premier notebook | Maintenant |
|---|---|
| Chemin de fichier propre à un ordinateur (`C:/Users/...`) | `spam.csv` à côté du notebook |
| Vocabulaire TF-IDF et échelles calculés sur toutes les données avant la séparation entraînement / test | Séparation en premier ; tout est appris sur l'entraînement uniquement |
| Un seul découpage, choix du modèle sur la précision | Validation croisée, F1, rappel, AUC, faux positifs et faux négatifs |
| Déséquilibre des classes constaté mais non traité | SMOTE dans l'entraînement, comparé à deux autres méthodes |
| Section « amélioration » qui ne réentraînait rien (colonnes identiques) | Vraies expériences, une variable à la fois |
| Sauvegarde avec `pickle` du vectoriseur et du modèle, sans l'échelle nécessaire | Pipeline complet sauvegardé avec `joblib`, rechargement testé hors du notebook |
| Messages avec « å£ » à la place de « £ » | Corrigé au chargement |
| Un email légitime long était classé spam à cause de sa longueur | Longueur plafonnée à 160 caractères |

Les chiffres du premier notebook (par exemple 98 % d'accuracy) ne sont pas comparables à ceux d'aujourd'hui : le protocole a changé.

## Limites

- Le modèle n'a appris que sur des SMS en anglais. Les messages en français ne sont pas fiables.
- Il n'a jamais été évalué sur de vrais emails. Nous avons essayé six emails écrits à la main : ils sont bien classés, le plus difficile (une facture) obtient un score de spam de 32,5 %. Six exemples ne constituent pas une évaluation.
- Le jeu de données est ancien : il contient peu de spams récents (hameçonnage, faux colis, cryptomonnaies).
- Le jeu de test ne contient que 131 spams. Manquer un spam de plus change le rappel de 0,8 point, et les écarts de 0,005 en F1 entre modèles ne sont pas significatifs.
- SMOTE n'est pas démontré meilleur que `class_weight` ou la duplication. Il crée des vecteurs de nombres, pas des messages lisibles.
- Le modèle exporté est réentraîné sur toutes les données : les résultats annoncés viennent de la version entraînée sur l'ensemble d'entraînement seul.
- Le modèle a retenu un défaut du jeu de données : la chaîne « &lt;#&gt; » apparaît dans 222 messages normaux et dans aucun spam, donc « gt » et « lt » comptent comme des indices de message normal.
- Le meilleur paramètre C de la régression logistique (100) est le plus grand de ceux testés. Nous n'avons pas cherché plus loin.

## Source des données

SMS Spam Collection, [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Almeida, T. A., Gómez Hidalgo, J. M. et Yamakami, A. (2011), *Contributions to the study of SMS spam filtering: new collection and results*.
