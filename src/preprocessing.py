import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def load_and_prepare_data():

    sgcc = pd.read_csv("sgcc_ml_ready.csv")
    pmu = pd.read_csv("merged_frequency_data.csv")

    y = sgcc["label"].values

    X = pd.concat([
        sgcc.drop(columns=["label"], errors="ignore").reset_index(drop=True),
        pmu.drop(columns=["label"], errors="ignore").reset_index(drop=True)
    ], axis=1)

    X = X.loc[:, ~X.columns.duplicated()]

    X = X.select_dtypes(include=[np.number])

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(
        X.median()
    ).values


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
