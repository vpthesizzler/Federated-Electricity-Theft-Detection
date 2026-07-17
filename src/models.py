import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE



def get_models():

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )


    xgb = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    return rf, xgb



def safe_proba(model, X):

    proba = model.predict_proba(X)

    if proba.shape[1] == 2:
        return proba[:,1]

    return np.zeros(len(X))



def train_with_smote(
    X_train,
    y_train,
    X_test
):

    sm = SMOTE(random_state=42)

    X_train, y_train = sm.fit_resample(
        X_train,
        y_train
    )


    rf, xgb = get_models()


    rf.fit(
        X_train,
        y_train
    )


    xgb.fit(
        X_train,
        y_train
    )


    prob = (
        safe_proba(rf, X_test)
        +
        safe_proba(xgb, X_test)
    ) / 2


    pred = (prob >= 0.5).astype(int)


    return pred, prob
