import numpy as np

from pandas.core.dtypes.common import is_numeric_dtype


def convert_all_cat_features_to_num_via_label_encoding(dataset):
    """
    Function to convert all categorical features in the dataset to numerical via label encoding.

    Input:
        dataset

    Output:
        dataset with all its features numerical
    """

    # Convert all categorical features in the dataset to numerical via label encoding
    for feature in dataset.columns.values:
        if not is_numeric_dtype(dataset[feature]):
            dataset[feature] = dataset[feature].astype('category').cat.codes

    dataset[dataset < 0] = np.nan  # If the value is negative then it is considered a NaN

    return dataset
