from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
confusion_matrix
)



def calculate_metrics(
    y_test,
    pred,
    prob
):

    results = {

        "accuracy":
        accuracy_score(y_test,pred),

        "precision":
        precision_score(
            y_test,
            pred,
            zero_division=0
        ),

        "recall":
        recall_score(
            y_test,
            pred,
            zero_division=0
        ),

        "f1":
        f1_score(
            y_test,
            pred,
            zero_division=0
        ),

        "roc_auc":
        roc_auc_score(
            y_test,
            prob
        )
    }


    return results
