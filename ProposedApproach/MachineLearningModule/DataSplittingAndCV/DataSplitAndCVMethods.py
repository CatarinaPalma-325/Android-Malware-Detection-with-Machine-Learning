from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut


# --------------- Random Split --------------- #
def random_split(x, y, test_size):
    """
    Function to randomly split the data into training and testing sets.

    Input:
        x - data matrix
        y - targets
        test_size - size of the testing set

    Output:
        x_train - training set
        x_test - testing set
        y_train - targets of the training set
        y_test - targets of the testing set
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test


# --------------- Stratified Random Split --------------- #
def stratified_random_split(x, y, test_size):
    """
    Function to randomly split (stratified) the data into training and testing sets.

    Input:
        x - data matrix
        y - targets
        test_size - size of the testing set

    Output:
        x_train - training set
        x_test - testing set
        y_train - targets of the training set
        y_test - targets of the testing set
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)
    return x_train, x_test, y_train, y_test


# --------------- Stratified 10-Fold Cross-validation --------------- #
def stratified_ten_fold_cv(x, y):
    """
    Function to obtain the indices to split data according to the stratified 10-fold cross-validation method.

    Input:
        x - data matrix
        y - targets

    Output:
        indices to split data
    """
    splits = StratifiedKFold(n_splits=10).split(x, y)
    return enumerate(splits)


# --------------- Leave-One-Out Cross-validation--------------- #
def leave_one_out_cv(x, y):
    """
    Function to obtain the indices to split data according to the leave-one-out cross-validation method.

    Input:
        x - data matrix
        y - targets

    Output:
        indices to split data
    """
    splits = LeaveOneOut().split(x, y)
    return enumerate(splits)
