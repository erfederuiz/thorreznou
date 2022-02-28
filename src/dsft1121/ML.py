from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, \
    mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston, load_iris
import pandas as pd
import numpy as np
########################
from sklearn.svm import SVC


def divide(dataset, target_column):
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    return X, y


def prepare_data(dataset, target_column, sampler_type=None, scaler_type='std', test_size=0.2):
    """
        Divides dataset in X and y, samples if necessary, divides in train and test and scales

        :param pandas dataset: Dataset we are working with
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


def model_scoring_classification(name, model, X_test, y_test, average="binary", multi_class="raise"):
    """
    Calculates model scoring for classification models
    :param str name: Column name
    :param model: Trained model to get metrics of
    :param X_test: X
    :param y_test: y
    :return: metrics
    """
    name = f'{name.upper()} (test data)'
    preds = model.predict(X_test)

    metrics = pd.DataFrame({name: [f'{accuracy_score(y_test, preds):.10f}',
                                   f'{precision_score(y_test, preds, average=average):.10f}',
                                   f'{recall_score(y_test, preds, average=average):.10f}',
                                   f'{f1_score(y_test, preds, average=average):.10f}']},
                           # f'{roc_auc_score(y_test, preds, multi_class=multi_class):.10f}']},
                           index=[['Accuracy (TP + TN/TT)', 'Precision (TP/TP + FP)', 'Recall (TP/TP + FN)',
                                   'F1 (har_mean Ac, Re)']])

    return metrics


def model_scoring_regression(name, model, X_test, y_test):
    """
    Calculates model scoring for regression models
    :param str name: Column name
    :param model: Trained model to get metrics of
    :param X_test: X
    :param y_test: y
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


def random_forest_classif(X_train, X_test, y_train, y_test):
    """
    Función para implementar un algoritmo de clasificación tipo RandomForestClassifier.
    El algoritmo se prueba con valores: n_estimator = 100, 200, 400, 600, 800; max_features = 2, 4 6
    Se hace un GridSearchCV con cv = 10 y de scoring la métrica "accuracy"

    Parámetros: X_train, y_train, X_test, y_test

    :return: el modelo entrenado, best_estimator, best_params y best_score basado en accuracy
    """
    rand_forest = RandomForestClassifier()

    rand_forest_param = {
        'n_estimators': [100, 200, 400, 600, 800],
        'max_features': [2, 4, 6]
    }

    grid = GridSearchCV(rand_forest, rand_forest_param, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

    model_fit = grid.fit(X_train, y_train)

    metrics = model_scoring_classification('RandomForest', model_fit, X_test, y_test)

    return model_fit, metrics, grid


def SVC1(X_train, X_test, y_train, y_test):
    """
    Función para implementar un algoritmo de clasificación tipo Support Vector Machine Classifier.
    El algoritmo se prueba con valores: C = np.arange(0.1, 0.9, 0.1); "gamma": scale, auto;
                                        "coef0": [-10,-1,0,0.1,0.5,1,10,100]; kernel = linear, poly, rbf
    Se hace un GridSearchCV con cv = 10 y de scoring la métrica "accuracy"

    :param X_train: X_train
    :param y_train: y_train
    :param X_test: X_test
    :param y_test: y_test

    :return: fitted model, grid and metrics
    """

    svc = SVC()

    svc_param = {
        "C": np.arange(0.1, 0.9, 0.1),
        "gamma": ["scale", "auto"],
        "coef0": [-10., -1., 0., 0.1, 0.5, 1, 10, 100],
        "kernel": ["linear", "poly", "rbf"],
    }

    grid = GridSearchCV(svc, svc_param, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

    model_fit = grid.fit(X_train, y_train)

    metrics = model_scoring_classification('SVC', model_fit, X_test, y_test, average="weighted", multi_class="ovo")

    return model_fit, metrics, grid


def LogistRegress(X_train, X_test, y_train, y_test):
    """
    Función para implementar un algoritmo de clasificación tipo LogisticRegression.
    El algoritmo se prueba con valores: C = np.arange(0.1, 4, 0.5); classifier__penalty: l1, l2
    Se hace un GridSearchCV con cv = 10 y de scoring la métrica "accuracy"

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


def best_classif_model(dataset, target_column, sampler_type, scaler_type, test_size):
    """
        Tests several models with given dataset and returns list with models, grids and a metrics dataset

        :param pandas dataset: Dataset we are working with
        :param str target_column: Target column name
        :param str sampler_type: Sampling to be applied. Accepts under, SMOTE, both and none
        :param str scaler_type: Scaler to be applied. Accepts std, minmax and None
        :param float test_size: Test size
        :return list: tuple list with X and y and sampled X and y
    """
    results = []
    metrics = pd.DataFrame()

    sets = prepare_data(dataset, target_column, sampler_type, scaler_type, test_size)

    for i in range(len(sets)):
        # RandomForestClassifier
        model, grid, model_metrics = random_forest_classif(sets[0], sets[1], sets[2], sets[3])
        results.append({
            'name': f'RandomForestClassifier {i}',
            'model': model,
            'grid': grid
        })
        pd.concat([metrics, model_metrics], axis=1)

        # SVC
        model, grid, model_metrics = SVC(sets[0], sets[1], sets[2], sets[3])
        results.append({
            'name': f'SVC {i}',
            'model': model,
            'grid': grid
        })
        pd.concat([metrics, model_metrics], axis=1)

        # LogisticRegression
        model, grid, model_metrics = SVC(sets[0], sets[1], sets[2], sets[3])
        results.append({
            'name': f'LogisticRegression {i}',
            'model': model,
            'grid': grid
        })
        pd.concat([metrics, model_metrics], axis=1)

        # KNN

    return results, metrics


def poly_reg(X_train, X_test, y_train, y_test, regular_type=None):
    """

    Apply a polynomial regression model with 3 polynomial levels

    :param X_train
    :param y_train
    :param X_test
    :param y_test
    :param regular_type: type of Regularization (ridge, lasso, elasticnet)
    :return model_fit, metrics: trained model and metrics
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
