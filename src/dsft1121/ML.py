from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


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


def model_scoring_classification(name, model, X_test, y_test):
    """
    Calculates model scoring for classification models
    :param str name: Column name
    :param model: Trained model to get metrics of
    :param x: X
    :param y: y
    :param set:
    :return: metrics
    """
    name = f'{name.upper()} (test data)'
    preds = model.predict(X_test)

    metrics = pd.DataFrame({name: [f'{accuracy_score(y_test, preds):.10f}',
                                   f'{precision_score(y_test, preds):.10f}',
                                   f'{recall_score(y_test, preds):.10f}',
                                   f'{f1_score(y_test, preds):.10f}',
                                   f'{roc_auc_score(y_test, preds):.10f}']},
                           index=[['Accuracy (TP + TN/TT)', 'Precision (TP/TP + FP)', 'Recall (TP/TP + FN)',
                                   'F1 (har_mean Ac, Re)', 'ROC AUC']])

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

    return model_fit, grid, metrics


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
        model, grid, model_metrics = random_forest_classif(sets[0], sets[1], sets[2], sets[3])
        results.append({
            'name': 'RandomForestClassifier',
            'model': model,
            'grid': grid
        })
        pd.concat([metrics, model_metrics], axis=1)

    return results, metrics
