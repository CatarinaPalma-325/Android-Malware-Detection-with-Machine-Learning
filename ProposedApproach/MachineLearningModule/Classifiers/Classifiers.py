import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


def get_classifier(ml_algorithm, x_train, y_train):
    """
    Function to get the ML classifier.

    Input:
        ml_algorithm - name of the chosen ML algorithm
        x_train - data matrix
        y_train - targets

    Output:
        classifier - trained ML model
    """

    if ml_algorithm == "RF":
        classifier = random_forest_fit(x_train, y_train)
    elif ml_algorithm == "SVM":
        classifier = svm_fit(x_train, y_train)
    elif ml_algorithm == "KNN":
        classifier = knn_fit(x_train, y_train)
    elif ml_algorithm == "NB":
        classifier = nb_fit(x_train, y_train)
    elif ml_algorithm == "MLP":
        classifier = mlp_fit(x_train, y_train)
    else:
        print("ERROR - No ML classifier choosen")
        sys.exit(1)

    return classifier


# --------------- Random Forest (RF) --------------- #
def random_forest_fit(x_train, y_train):
    """
    Function to fit the model to data matrix x and targets y and obtain a trained Random Forest (RF) model.

    Input:
        x_train - data matrix
        y_train - targets

    Output:
        trained RF model
    """
    return RandomForestClassifier().fit(x_train, y_train)


# --------------- Support Vector Machine (SVM) --------------- #
def svm_fit(x_train, y_train):
    """
    Function to fit the model to data matrix x and targets y and obtain a trained Support Vector Machine (SVM) model.

    Input:
        x_train - data matrix
        y_train - targets

    Output:
        trained SVM model
    """
    return svm.SVC().fit(x_train, y_train)


# --------------- K-Nearest Neighbours (KNN) --------------- #
def knn_fit(x_train, y_train):
    """
    Function to fit the model to data matrix x and targets y and obtain a trained K-Nearest Neighbours (KNN) model.

    Input:
        x_train - data matrix
        y_train - targets

    Output:
        trained KNN model
    """
    return KNeighborsClassifier().fit(x_train, y_train)


# --------------- Naive Bayes (NB) --------------- #
def nb_fit(x_train, y_train):
    """
    Function to fit the model to data matrix x and targets y and obtain a trained Naive Bayes (NB) model.

    Input:
        x_train - data matrix
        y_train - targets

    Output:
        trained (Multinomial) NB model
    """
    return MultinomialNB().fit(x_train, y_train)


# --------------- Multi-layer Perceptron (MLP) --------------- #
def mlp_fit(x_train, y_train):
    """
    Function to fit the model to data matrix x and targets y and obtain a trained Multi-layer Perceptron (MLP) model.

    Input:
        x_train - data matrix
        y_train - targets

    Output:
        trained MLP model
    """
    return MLPClassifier().fit(x_train, y_train)
