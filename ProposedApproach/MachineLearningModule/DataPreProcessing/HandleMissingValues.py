

# --------------- Removing features with missing values --------------- #
def remove_features_with_missing_values(dataset):
    """
    Function that removes all dataset features containing missing values.

    Input:
        dataset with missing values

    Output:
        dataset without features containing missing values
    """
    return dataset.dropna(axis=1)


# --------------- Removing instances with missing values --------------- #
def remove_instances_with_missing_values(dataset):
    """
    Function that removes all dataset instances containing missing values.

    Input:
        dataset with missing values

    Output:
        dataset without instances containing missing values
    """
    dataset = remove_all_na_features(dataset)
    return dataset.dropna()


# --------------- Imputing missing values with the feature mean --------------- #
def impute_missing_values_with_feature_mean(dataset):
    """
    Function that imputes missing values with the feature mean.

    Input:
        dataset with missing values

    Output:
        dataset with all previously missing values imputed with the feature mean
    """
    dataset = remove_all_na_features(dataset)  # Helper function to remove all features with all values NaN

    dataset = dataset.fillna(dataset.mean(numeric_only=True))
    return dataset


# --------------- Imputing missing values with the feature median --------------- #
def impute_missing_values_with_feature_median(dataset):
    """
    Function that imputes missing values with the feature median.

    Input:
        dataset with missing values

    Output:
        dataset with all previously missing values imputed with the feature median
    """
    dataset = remove_all_na_features(dataset)  # Helper function to remove all features with all values NaN

    dataset = dataset.fillna(dataset.median(numeric_only=True))
    return dataset


# --------------- Imputing missing values with the feature mode --------------- #
def impute_missing_values_with_feature_mode(dataset):
    """
    Function that imputes missing values with the feature mode.

    Input:
        dataset with missing values

    Output:
        dataset with all previously missing values imputed with the feature mode
    """
    dataset = remove_all_na_features(dataset)  # Helper function to remove all features with all values NaN

    dataset = dataset.fillna(dataset.mode().iloc[0])
    return dataset


# --------------- Helper function - Remove all features with all values NaN ---------------
def remove_all_na_features(dataset):
    """
    Function that removes all features that have all their values NaN.

    Input:
        dataset

    Output:
        dataset without features all NaN
    """
    for f in dataset.columns.values:
        if dataset[f].isna().sum()/len(dataset) == 1:
            dataset = dataset.drop(f, axis=1)
    return dataset
