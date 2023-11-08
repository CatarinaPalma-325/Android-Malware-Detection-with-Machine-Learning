import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def min_max_normalisation(dataset):
    """
    Function to perform Min-max normalisation.

    Input:
        dataset

    Output:
        dataset with all features normalised via Min-max normalisation
    """
    features = list(dataset.columns.values)
    scaler = MinMaxScaler()  # By default, normalisation is to the range [0, 1]
    dataset = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(dataset, columns=features)

    return dataset
