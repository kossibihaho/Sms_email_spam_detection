from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE as RS


def get_models() -> dict:
    return {
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
        "ComplementNB": ComplementNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RS),
        "LinearSVC": LinearSVC(random_state=RS),
        "SVC_rbf": SVC(kernel="rbf", random_state=RS),
        "KNN": KNeighborsClassifier(metric="cosine", algorithm="brute"),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=RS),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=RS, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RS),
        "Bagging": BaggingClassifier(n_estimators=50, random_state=RS, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RS),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=RS, eval_metric="logloss", n_jobs=-1),
    }
