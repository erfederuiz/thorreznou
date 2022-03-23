from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, \
    mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


def divide(dataset, target_column):
    """
    Divides the dataset in X and y.
    This function is called inside prepare_data function (see below) in order to prepare data.

    :param dataset:
    :param target_column:
    :return: X and y
    """
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    return X, y


def prepare_data(dataset, target_column, sampler_type=None, scaler_type='std', test_size=0.2):
    """
        Creates a list with several train-test sets to test with:

        1. Divides the dataset in X and y
        2. If scaler_type is under/smote/both, it samples the data. It will append to the list the generated train-test sets
        3. After that, it will scale all sets
        4. Returns A LIST OF LISTS with the following format:
            [
                [X_train, X_test, y_train, y_test] #set original
                [X_train, X_test, y_train, y_test] #Solo si se samplean los datos
            ]

        ¡IMPORTANT! The output of this function is used as input in all the model functions


        EXAMPLE:

        If we only want to use the original set, we can just use the first set of the array.
        Let's create a linear regression with library functions:
            sets = prepare_data(dataset, 'target')
            model, metrics = lin_reg(sets[0][0], sets[0][1], sets[0][2], sets[0][3])

        If sample is needed, this function will return more than one set. In that case, we will need to iterate over it:
            sets = prepare_data(dataset, 'target', sample_type='SMOTE')
            for i in range(len(sets)):
                model, metrics = lin_reg(sets[i][0], sets[i][1], sets[i][2], sets[i][3])


        :param pandas.__class__ dataset: Dataset we are working with
        :param str target_column: Target column name
        :param str sampler_type: Sampling to be applied. Accepts under, SMOTE, both and none
        :param str scaler_type: Scaler to be applied. Accepts std, minmax and None
        :param float test_size: Test size
        :return list: tuple list with X and y and sampled X and y
    """
    sets = []
    splited_sets = []
    scaled_sets = []

    # 1. Divide in X and y
    X, y = divide(dataset, target_column)
    sets.append((X, y))

    # 2. Sample if necessary
    if sampler_type == 'under':
        X_unders, y_unders = RandomUnderSampler(random_state=42).fit_resample(X, y)
        sets.append((X_unders, y_unders))

    elif sampler_type == 'SMOTE':
        X_overs, y_overs = SMOTE(random_state=42).fit_resample(X, y)
        sets.append((X_overs, y_overs))

    elif sampler_type == 'both':
        X_unders, y_unders = RandomUnderSampler(random_state=42).fit_resample(X, y)
        sets.append((X_unders, y_unders))
        X_overs, y_overs = SMOTE(random_state=42).fit_resample(X, y)
        sets.append((X_overs, y_overs))

    for set in sets:
        X_train, X_test, y_train, y_test = train_test_split(set[0], set[1], test_size=test_size,
                                                            random_state=42)
        splited_sets.append([X_train, X_test, y_train, y_test])

    # Optimize memory usage
    del sets

    for set in splited_sets:
        if scaler_type == "std":
            std_scale = StandardScaler().fit(set[0])  # fit(X_train)
            X_train_std = std_scale.transform(set[0])  # fit(X_train)
            X_test_std = std_scale.transform(set[1])  # fit(X_test)
            scaled_sets.append([X_train_std, X_test_std, set[2], set[3]])

        elif scaler_type == "minmax":
            minmax_scaler = MinMaxScaler().fit(set[0])  # fit(X_train)
            X_train_scl = minmax_scaler.transform(set[0])  # fit(X_train)
            X_test_scl = minmax_scaler.transform(set[1])  # fit(X_test)
            scaled_sets.append([X_train_scl, X_test_scl, set[2], set[3]])
        else:
            scaled_sets = splited_sets

    return scaled_sets


def model_scoring_regression(name, model, X_test, y_test):
    """
    Calculates model scoring dataframe for regression models
    This function is called inside regression model functions (see below) in order to generate metrics dataset.

    :param str name: Column name
    :param model: Trained model to get metrics of
    :param X_test: X_train
    :param y_test: y_test
    :return: metrics
    """
    name = f'{name.upper()} (test data)'
    preds = model.predict(X_test)

    metrics = pd.DataFrame({name: [f'{model.score(X_test, y_test):.10f}',
                                   f'{mean_absolute_error(y_test, preds):.10f}',
                                   f'{mean_absolute_percentage_error(y_test, preds):.10f}',
                                   f'{mean_squared_error(y_test, preds):.10f}',
                                   f'{np.sqrt(mean_squared_error(y_test, preds)):.10f}']},
                           index=[['Score (R2 coef.)', 'MAE', 'MAPE', 'MSE', 'RMSE']])

    return metrics


def lin_reg(X_train, X_test, y_train, y_test, regular_type=None):
    """
    Apply a lineal regression model with 3 polynomial levels
    :param X_train
    :param y_train
    :param X_test
    :param y_test
    :param regular_type: type of Regularization (ridge, lasso, elasticnet)
    :return trained model and metrics dataframe
    """

    model_fit = LinearRegression()

    model_fit.fit(X_train, y_train)

    metrics = model_scoring_regression('LinearRegression', model_fit, X_test, y_test)

    # Regularization if needed

    if regular_type == 'ridge':
        ridgeR = Ridge(alpha=10)
        ridgeR.fit(X_train, y_train)
        metrics = pd.concat([metrics,
                             model_scoring_regression('LINREG-RIDGE', ridgeR, X_test, y_test)], axis=1)

    elif regular_type == 'lasso':
        lassoR = Lasso(alpha=1)
        lassoR.fit(X_train, y_train)
        metrics = pd.concat([metrics,
                             model_scoring_regression('LINREG-LASSO', lassoR, X_test, y_test)], axis=1)

    elif regular_type == 'elasticnet':
        elastic_net = ElasticNet(alpha=1, l1_ratio=0.5)
        elastic_net.fit(X_train, y_train)
        metrics = pd.concat([metrics,
                             model_scoring_regression('LINREG-ELASTICNET', elastic_net, X_test, y_test)], axis=1)

    elif regular_type == 'None':

        print("Regularization not necessary")

    return model_fit, metrics


def poly_reg(X_train, X_test, y_train, y_test, regular_type=None):
    """

    Apply a polynomial regression model with 2 polynomial levels

    :param X_train
    :param y_train
    :param X_test
    :param y_test
    :param regular_type: type of Regularization (ridge, lasso, elasticnet)
    :return trained model and metrics dataframe
    """

    poly_feats = PolynomialFeatures(degree=2)

    X_poly = poly_feats.fit_transform(X_train)

    model_fit = LinearRegression()

    model_fit.fit(X_poly, y_train)

    X_poly_test = poly_feats.transform(X_test)

    metrics = model_scoring_regression('PolynomialRegression', model_fit, X_poly_test, y_test)

    # Regularization if needed

    if regular_type == 'ridge':
        ridgeR = Ridge(alpha=10)
        ridgeR.fit(X_train, y_train)
        metrics = pd.concat([metrics,
                             model_scoring_regression('POLREG-RIDGE', ridgeR, X_test, y_test)], axis=1)

    elif regular_type == 'lasso':
        lassoR = Lasso(alpha=1)
        lassoR.fit(X_train, y_train)
        metrics = pd.concat([metrics,
                             model_scoring_regression('POLREG-LASSO', lassoR, X_test, y_test)], axis=1)

    elif regular_type == 'elasticnet':
        elastic_net = ElasticNet(alpha=1, l1_ratio=0.5)
        elastic_net.fit(X_train, y_train)
        metrics = pd.concat([metrics,
                             model_scoring_regression('POLREG-ELASTICNET', elastic_net, X_test, y_test)], axis=1)

    elif regular_type == 'None':

        print("Regularization not necessary")

    return model_fit, metrics


def model_scoring_classification(name, model, X_test, y_test, average="binary"):
    """
    Calculates model scoring for classification models
    This function is called inside model classification functions (see below) in order to generate metrics dataset.

    :param str name: Column name
    :param model: Trained model to get metrics of
    :param X_test: X_test
    :param y_test: y_test
    :param average: average param for precision, recall and score. If multiclass, use "weighted"
    :param multi_class: multi_class for roc_auc_score (ovr, ovo)
    :return: metrics dataframe
    """
    name = f'{name.upper()} (test data)'
    preds = model.predict(X_test)

    metrics = pd.DataFrame({name: [f'{accuracy_score(y_test, preds):.10f}',
                                   f'{precision_score(y_test, preds, average=average):.10f}',
                                   f'{recall_score(y_test, preds, average=average):.10f}',
                                   f'{f1_score(y_test, preds, average=average):.10f}']},
                           index=[['Accuracy (TP + TN/TT)', 'Precision (TP/TP + FP)', 'Recall (TP/TP + FN)',
                                   'F1 (har_mean Ac, Re)']])

    return metrics


def random_forest_classif(X_train, X_test, y_train, y_test):
    """
    Tests several parameters with a RandomForestClassifier model
    GridSearchCV is used with cv=10 and accuracy metric

    :param X_train: X_train
    :param X_test: X_test
    :param y_train: y_train
    :param y_test: y_test

    :returns: trained model, metrics dataframe and grid used
    """
    rand_forest = RandomForestClassifier()

    rand_forest_param = {
        'n_estimators': [100, 200, 400, 600, 800],
        'max_features': [2, 4, 6]
    }

    grid = GridSearchCV(rand_forest, rand_forest_param, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

    model_fit = grid.fit(X_train, y_train)

    metrics = model_scoring_classification('RandomForest', model_fit, X_test, y_test, average="weighted")

    return model_fit, metrics, grid


def SVC_model(X_train, X_test, y_train, y_test):
    """
    Tests several parameters with a SVC model
    GridSearchCV is used with cv=10 and accuracy metric

    :param X_train: X_train
    :param y_train: y_train
    :param X_test: X_test
    :param y_test: y_test

    :returns: trained model, metrics dataframe and grid used
    """

    svc = SVC()

    svc_param = {
        "C": np.arange(0.1, 1, 0.3),
        "gamma": ["scale", "auto"],
        "coef0": [-10., -1., 10],
        "kernel": ["linear", "poly", "rbf"],
    }

    grid = GridSearchCV(svc, svc_param, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

    model_fit = grid.fit(X_train, y_train)

    metrics = model_scoring_classification('SVC', model_fit, X_test, y_test, average="weighted")

    return model_fit, metrics, grid


def LogistRegress(X_train, X_test, y_train, y_test):
    """
    ests several parameters with a LogisticRegression model
    GridSearchCV is used with cv=10 and accuracy metric


    :param X_train: X_train
    :param y_train: y_train
    :param X_test: X_test
    :param y_test: y_test

    :return: fitted model, grid and metrics
    """

    log_reg = LogisticRegression()

    log_param = {
        'penalty': ['l2'],
        'C': np.arange(0.1, 4, 0.5),
        'solver': ['liblinear', 'newton-cg', 'lbfgs']
    }

    grid = GridSearchCV(log_reg, log_param, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

    model_fit = grid.fit(X_train, y_train)

    metrics = model_scoring_classification('LogisticRegression', model_fit, X_test, y_test, average="weighted")

    return model_fit, metrics, grid


# FUNCIÓN AUXILIAR PARA, DE UNA LISTA (EN PRINCIPIO, SIEMPRE UNA LISTA CON SCORE), PODER ESCOGER EL ELEMENTO MÁS ALTO
def change_score(element, p=0.90):
    """
    Function that iterates through the values parameter elements, converting them to positive (if necessary) and discarding those higher than the
    parameter value "p" (typically 0.90).
    It is useful to help the "knn" function (and the "corr_target" auxiliary function), with the aim of being able to identify the best score, and
    also discarding high values through the "p" parameter and thus avoiding OVERFITTING.
    :param element: List with elements (scores or correlations) to fit
    :param p: if the user prefer, percent to discard the high score or correlation. Dont could be higher than 1.
    :return: list with the changed values
    """
    if p >= 1:
        p = 0.90
    lista = []
    for i in element:
        if i < 0:
            i = - i
        if i >= 0.90:
            i = i - i
        lista.append(i)
    return lista


# FUNCIÓN PARA OBTENER LAS COLUMNAS CON MEJOR CORRELACIÓN CON EL TARGET
def corr_target(X, y, n_columns=1):
    """
    Function that identifies the two columns (max 4, if the user prefers) with the best correlation with respect to the target, of all Dataset.
    If the target is categorical, search the two columns with the best correlation between them.
    :param X: X_train from get the name of the best columns from correlation
    :param y: y_train
    :param n_columns: if the user prefer, numbers of columns to get from the correlation. Don't could be less than 1, higher than 4 or more than the
                      total of Dataset.
    :return: list with the name of the best columns from correlation. The first element always be the target, the second the column with higher
             correlation, the third the second one with higher correlation...
    """
    if n_columns <= 0:
        n_columns = 1

    df = X.join(y)

    lista_final = []

    if n_columns > len(df.columns):
        n_columns = len(df.columns)

    correlation_mat = df.corr()
    corr_pairs = correlation_mat.unstack()

    n = 0
    lista = []
    for i in corr_pairs.index:
        if i[0] == df.columns[-1]:
            lista.append(n)
        n = n + 1

    if len(lista) <= 0:  # <---Only in case the target is categorical
        correlation_mat = df.corr()
        corr_pairs = correlation_mat.unstack()
        lista = []
        lista = change_score(corr_pairs)
        mayor = max(lista)
        indice = lista.index(mayor)
        x = corr_pairs.index[indice]
        lista_final.append(df.columns[-1])
        lista_final.append(x[0])
        lista_final.append(x[1])
        return lista_final

    corr_pairs = corr_pairs[(lista[0]):(lista[-1] + 1)]

    lista_2 = []
    lista_2 = change_score(corr_pairs)

    lista_3 = lista_2.copy()
    lista_3.sort(reverse=True)

    if n_columns >= 4:
        mayor_4 = lista_3[3]
        indice_4 = lista_2.index(mayor_4)
        d = corr_pairs.index[indice_4]
        lista_final.append(d[1])

    if n_columns >= 3:
        mayor_3 = lista_3[2]
        indice_3 = lista_2.index(mayor_3)
        c = corr_pairs.index[indice_3]
        lista_final.append(c[1])

    if n_columns >= 2:
        mayor_2 = lista_3[1]
        indice_2 = lista_2.index(mayor_2)
        b = corr_pairs.index[indice_2]
        lista_final.append(b[1])

    if n_columns >= 1:
        mayor_1 = lista_3[0]
        indice_1 = lista_2.index(mayor_1)
        a = corr_pairs.index[indice_1]
        lista_final.append(a[1])
        lista_final.append(a[0])

    lista_final.reverse()

    return lista_final


# FUNCÓN PARA KNN
def knn(X_train, X_test, y_train, y_test, K=0, n_columns=2):
    """
    Function to implement a classification algorithm such as KNeighborsClassifier.
    :param X_train: X_train
    :param X_test: X_test
    :param y_train: y_train
    :param y_test: y_test
    :param K: if the user prefer, numbers of KNeighbors to fit the classifier
    :para n_columns: if the user prefer, number of columns to get the correlation to target. Only to "corr_target" auxiliary function
    :return: fitted model and metrics
    In case the Dataframe does not fit the classifier, returns None
    """
    try:
        if len(X_train.columns) < 2:
            print("Not enough columns")
            return None

        if len(X_train.columns) > 2:
            if n_columns < 2:
                n_columns = 2
            corr = corr_target(X_train, y_train, n_columns)
            colum1 = corr[1]
            colum2 = corr[2]

            X_train = pd.concat([X_train[colum1], X_train[colum2]], axis=1)
            X_test = pd.concat([X_test[colum1], X_test[colum2]], axis=1)

        scores_train = []
        scores_test = []
        k_range = range(1, 20)

        if K == 0:
            for i in k_range:
                knn_f = KNeighborsClassifier(n_neighbors=i)
                knn_f.fit(X_train, y_train)
                scores_train.append(knn_f.score(X_train, y_train))
                scores_test.append(knn_f.score(X_test, y_test))

            scores_change = change_score(scores_test)

            try:
                k = scores_test.index(max(scores_change)) + 1
            except:
                scores_test.reverse()
                k = scores_test.index(min(scores_test)) + 1
        else:
            if K < 0:
                K = -K
            k = K

        knn = KNeighborsClassifier(n_neighbors=k)
        model = knn.fit(X_train, y_train)

        metrics = model_scoring_classification('KNN', model, X_test, y_test, average="weighted")

        return model, metrics
    except ValueError:
        print("Dataframe does not fit classifier")
        return None
