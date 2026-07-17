import os
import pandas as pd


def create_directories():

    os.makedirs(
        "results/tables",
        exist_ok=True
    )

    os.makedirs(
        "results/plots",
        exist_ok=True
    )



def save_results(results):

    df = pd.DataFrame(results)

    df.to_csv(
        "results/tables/smote_comparison.csv",
        index=False
    )
