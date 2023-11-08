import statistics
import sys
import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine


# --------------- Relevance-Redundancy Feature Selection (RRFS) --------------- #
def relevance_redundancy_feature_selection(dataset, class_name, relevance_measure, ms):
    """
    Function to perform Relevance-Redundancy Feature Selection (RRFS).

    Input:
        dataset
        class name
        relevance_measure - chosen relevance measure FR (Fisher's Ratio) or MM (Mean-Median)
        ms - maximum allowed similarity between consecutive pairs of features

    Output:
        dataset after RRFS
    """

    # Relevance
    if relevance_measure == "FR":
        dataset = relevance_measure_fisher_ratio(dataset, class_name)
    elif relevance_measure == "MM":
        dataset = relevance_measure_mean_median(dataset)
    else:
        print("ERROR - No feature selection relevance method choosen")
        sys.exit(1)

    # Redundancy
    dataset = redundancy_measure_absolute_cosine(dataset, ms)

    return dataset


# --------------- Relevance measure (unsupervised): Mean-median (MM) --------------- #
def relevance_measure_mean_median(dataset):
    """
    Function to reduce the number of features in the dataset based on the Mean-median (MM) relevance measure.

    Input:
        dataset

    Output:
        dataset with the features kept after applying the (unsupervised) relevance measure MM
    """

    features = dataset.columns.values
    arr_mm, arr_mm_features = [], []

    for feature in features:
        mm_value = abs(np.mean(dataset[feature]) - np.median(dataset[feature]))
        arr_mm.append(mm_value)
        arr_mm_features.append(feature)

    df = pd.DataFrame(arr_mm, arr_mm_features)
    features = cumulative_relevance(df)

    return dataset[features]


# --------------- Relevance measure (supervised): Fisher's ratio (FR) --------------- #
def relevance_measure_fisher_ratio(dataset, class_name):
    """
    Function to reduce the number of features in the dataset based on the Fisher's ratio (FR) relevance measure.

    Input:
        dataset

    Output:
        dataset with the features kept after applying the (supervised) relevance measure FR
    """

    features = dataset.columns.values
    arr_fr, arr_fr_features = [], []

    class_name_values = dataset[class_name].unique()

    for feature in features:

        # Mean values for each class label
        feature_mean_for_class0 = np.mean(
            dataset.loc[dataset.index[dataset[class_name] == class_name_values[0]].tolist()][feature])
        feature_mean_for_class1 = np.mean(
            dataset.loc[dataset.index[dataset[class_name] == class_name_values[1]].tolist()][feature])

        # Variance values for each class label
        feature_variance_for_class0 = statistics.variance(
            dataset.loc[dataset.index[dataset[class_name] == class_name_values[0]].tolist()][feature])
        feature_variance_for_class1 = statistics.variance(
            dataset.loc[dataset.index[dataset[class_name] == class_name_values[1]].tolist()][feature])

        if feature_variance_for_class0 + feature_variance_for_class1 == 0:
            continue

        fr_value = np.square(feature_mean_for_class0 - feature_mean_for_class1) / (
                feature_variance_for_class0 + feature_variance_for_class1)
        arr_fr.append(fr_value)
        arr_fr_features.append(feature)

    df = pd.DataFrame(arr_fr, arr_fr_features)
    features = cumulative_relevance(df)

    return dataset[features]


# --------------- Cumulative relevance --------------- #
def cumulative_relevance(df):
    """
    Function to obtained the features to be kept in the dataset based on cumulative relevance.

    Input:
        df - dataframe containing the relevance measure values for each feature and each feature's name

    Output:
        features to be kept based on the cumulative relevance threshold
    """
    total = df[0].sum()
    for i in df.index.values:
        df.loc[i][0] = df.loc[i][0] / total

    df = df.sort_values(by=0, ascending=False)

    cumulative = 0
    features = []
    for i in df.index.values:
        if cumulative < 0.99:
            cumulative = cumulative + df.loc[i][0]
            features.append(df.loc[i].name)

    return features


# --------------- Redundancy measure: Absolute cosine (AC) --------------- #
def redundancy_measure_absolute_cosine(dataset, ms):
    """
    Function to reduce the number of features in the dataset based on the Absolute cosine (AC) redundancy measure.

    Input:
        dataset

    Output:
        dataset with the features kept after applying the redundancy measure AC
    """

    features = list(dataset.columns.values)

    features_keep = [features[0]]
    prev = 1

    for i in range(2, len(features)):
        # Cosine between consecutive features
        s = 1 - cosine(dataset[features[i]], dataset[features[prev]])
        if s < ms:
            features_keep.append(features[i])
            prev = i

    return dataset[features_keep]
