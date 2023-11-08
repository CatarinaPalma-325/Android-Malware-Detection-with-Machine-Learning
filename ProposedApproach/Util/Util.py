import numpy as np


def print_evaluation_metrics(evaluation_metrics):
    """
    Function to print the mean and distance deviation values of each evaluation metric.

    Input:
        all the obtained evaluation metrics
    """

    means = get_evaluation_metrics_means(evaluation_metrics)

    print("\n------------------ Model Evaluation Metrics ------------------\n")
    print("Accuracy: ", means[0])
    print("Confusion Matrix: ", "| TN:", means[1][0], "| FP:", means[1][1], "| FN:",
          means[1][2], "| TP:", means[1][3], "|")
    print("Precision: ", means[2])
    print("Recall: ", means[3])
    print("F1-score: ", means[4])
    print("ROC-AUC: ", means[5])
    print("\n-------------------------------------------------------------\n")

    distance_deviations = get_evaluation_metrics_means(evaluation_metrics)

    print("\n--------------------- Distance Deviations ---------------------\n")
    print("Accuracy: ", distance_deviations[0])
    print("Confusion Matrix: ", "| TN:", distance_deviations[1], "| FP:", distance_deviations[2], "| FN:",
          distance_deviations[3], "| TP:", distance_deviations[4], "|")
    print("Precision: ", distance_deviations[5])
    print("Recall: ", distance_deviations[6])
    print("F1-score: ", distance_deviations[7])
    print("ROC-AUC: ", distance_deviations[8])
    print("\n-------------------------------------------------------------\n")


def get_evaluation_metrics_means(evaluation_metrics):
    """
    Function to get the mean values for each evaluation metric.

    Input:
        all the obtained evaluation metrics

    Output:
        mean values for each evaluation metric
    """

    accuracies = evaluation_metrics[0]
    confusion_matrices = evaluation_metrics[1]
    precisions = evaluation_metrics[2]
    recalls = evaluation_metrics[3]
    f1_scores = evaluation_metrics[4]
    roc_aucs = evaluation_metrics[5]

    means = [
        round(sum(accuracies) / len(accuracies), 4) * 100,
        sum(confusion_matrices) / len(confusion_matrices),
        round(sum(precisions) / len(precisions), 4) * 100,
        round(sum(recalls) / len(recalls), 4) * 100,
        round(sum(f1_scores) / len(f1_scores), 4) * 100,
        round(sum(roc_aucs) / len(roc_aucs), 4) * 100
    ]

    return means


def get_evaluation_metrics_distance_deviations(evaluation_metrics):
    """
    Function to get the distance deviation value for each evaluation metric.

    Input:
        all the obtained evaluation metrics

    Output:
        distance deviation values for each evaluation metric
    """

    accuracies = evaluation_metrics[0]
    confusion_matrices = evaluation_metrics[1]
    precisions = evaluation_metrics[2]
    recalls = evaluation_metrics[3]
    f1_scores = evaluation_metrics[4]
    roc_aucs = evaluation_metrics[5]

    cm_tn, cm_fp, cm_fn, cm_tp = [], [], [], []
    for x in range(3):
        cm_tn.append((confusion_matrices[x])[0])
        cm_fp.append((confusion_matrices[x])[1])
        cm_fn.append((confusion_matrices[x])[2])
        cm_tp.append((confusion_matrices[x])[3])

    distance_deviations = [
        round(np.sqrt(np.var(accuracies)), 4) * 100,
        np.sqrt(np.var(cm_tn)),
        np.sqrt(np.var(cm_fp)),
        np.sqrt(np.var(cm_fn)),
        np.sqrt(np.var(cm_tp)),
        round(np.sqrt(np.var(precisions)), 4) * 100,
        round(np.sqrt(np.var(recalls)), 4) * 100,
        round(np.sqrt(np.var(f1_scores)), 4) * 100,
        round(np.sqrt(np.var(roc_aucs)), 4) * 100
    ]

    return distance_deviations
