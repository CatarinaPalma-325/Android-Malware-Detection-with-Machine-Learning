import sklearn.metrics as metrics


# --------------- Accuracy --------------- #
def accuracy(y_true, y_pred):
    """
    Function to obtain the accuracy score.

    Input:
        y_true - correct labels
        y_pred - predicted labels

    Output:
        accuracy score
    """
    return metrics.accuracy_score(y_true, y_pred)


# --------------- Confusion Matrix --------------- #
def confusion_matrix(y_true, y_pred):
    """
    Function to obtain the confusion matrix.

    Input:
        y_true - correct labels
        y_pred - predicted labels

    Output:
        confusion matrix
    """
    return metrics.confusion_matrix(y_true, y_pred).ravel()


# --------------- Precision --------------- #
def precision(y_true, y_pred):
    """
    Function to obtain the precision score.

    Input:
        y_true - correct labels
        y_pred - predicted labels

    Output:
        precision score
    """
    return metrics.precision_score(y_true, y_pred)


# --------------- Recall --------------- #
def recall(y_true, y_pred):
    """
    Function to obtain the recall score.

    Input:
        y_true - correct labels
        y_pred - predicted labels

    Output:
        recall score
    """
    return metrics.recall_score(y_true, y_pred)


# --------------- F1-score --------------- #
def f1_score(y_true, y_pred):
    """
    Function to obtain the F1-score.

    Input:
        y_true - correct labels
        y_pred - predicted labels

    Output:
        F1-score
    """
    return metrics.f1_score(y_true, y_pred)


# --------------- Area Under the Curve - Receiver Operating Characteristic (ROC-AUC) --------------- #
def roc_auc(y_true, y_pred):
    """
    Function to obtain the ROC-AUC score

    Input:
        y_true - correct labels
        y_pred - predicted labels

    Output:
        ROC-AUC score
    """
    return metrics.roc_auc_score(y_true, y_pred)
