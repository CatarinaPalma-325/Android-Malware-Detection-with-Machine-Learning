from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


# --------------- Random Undersampling --------------- #
def random_undersampling(x, y):
    """
    Function to perform random undersampling.

    Input:
        x - data matrix
        y - targets

    Output:
        x - data matrix after random undersampling
        y - targets after random undersampling
    """
    x_res, y_res = RandomUnderSampler(random_state=42).fit_resample(x, y)

    return x_res, y_res


# --------------- Random Oversampling --------------- #
def random_oversampling(x, y):
    """
    Function to perform random oversampling.

    Input:
        x - data matrix
        y - targets

    Output:
        x - data matrix after random oversampling
        y - targets after random oversampling
    """
    x_res, y_res = RandomOverSampler(random_state=42).fit_resample(x, y)

    return x_res, y_res


# --------------- Synthetic Minority Over-sampling TEchnique (SMOTE) --------------- #
def smote(x, y):
    """
    Function to perform Synthetic Minority Over-sampling TEchnique (SMOTE) with the 'minority' sampling strategy.

    Input:
        x - data matrix
        y - targets

    Output:
        x - data matrix after SMOTE
        y - targets after SMOTE
    """
    x_res, y_res = SMOTE(sampling_strategy='minority').fit_resample(x, y)

    return x_res, y_res
