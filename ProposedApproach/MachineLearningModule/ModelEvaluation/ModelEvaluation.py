import numpy as np

from MachineLearningModule.ModelEvaluation import EvaluationMetrics


def model_evaluation(y_test, y_pred):
    """
    Function to obtain various evaluation metrics.

    Input:
        y_true - correct labels
        y_pred - predicted labels

    Output:
        evaluation metrics (accuracy, confusion_matrix, precision, recall, F1-score and ROC-AUC)
    """

    accuracy = EvaluationMetrics.accuracy(y_test, y_pred)
    confusion_matrix = EvaluationMetrics.confusion_matrix(y_test, y_pred)
    precision = EvaluationMetrics.precision(y_test, y_pred)
    recall = EvaluationMetrics.recall(y_test, y_pred)
    f1_score = EvaluationMetrics.f1_score(y_test, y_pred)

    roc_auc = None
    if len(np.unique(y_test)) > 1:
        roc_auc = EvaluationMetrics.roc_auc(y_test, y_pred)

    return accuracy, confusion_matrix, precision, recall, f1_score, roc_auc
