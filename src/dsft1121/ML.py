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
    :param average: average for precision, recall and score
    :param multi_class: multi_class for roc_auc_score (ovr, ovo)
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

    metrics = model_scoring_classification('RandomForest', model_fit, X_test, y_test, average="weighted")

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


def change_score(element):
    lista = []
    for i in element:
        if i < 0:
            i = - i
        if i >= 0.90:
            i = i - i
        lista.append(i)
    return lista


# FUNCIÓN PARA OBTENER LAS COLUMNAS CON MEJOR CORRELACIÓN CON EL TARGET
def corr_target(X, y, n_columns=1):  # Los parámetros introducidos son "X", "y" y el número de columnas (con mayor correlación con el target) deseadas
    if n_columns <= 0:  # Para impedir que que se escoja un número menor a 1
        n_columns = 1

    df = X.join(y)  # Unir la X con la y para llevar a cabo la correlación

    lista_final = []  # Esta es la lista que guardaría los nombres del targer y las columnas más correlacionadas

    if n_columns > len(df.columns):  # Comprobar que el número de columnas escogidas no sea mayor que el número disponible
        n_columns = len(df.columns)

    correlation_mat = df.corr()  # Obtener la correlación
    corr_pairs = correlation_mat.unstack()  # Obtener un Series Multiíndice con todas las correlacione y sus valores

    n = 0
    lista = []
    for i in corr_pairs.index:  # Indentifiar la posición de aquellos índices que correspondan con las correlaciones con el target
        if i[0] == df.columns[-1]:
            lista.append(n)
        n = n + 1

    if len(lista) <= 0:  # EN CASO DE QUE SE TRATE DE UN TARGET CATEGÓRICO, Y NO PUEDA SER CORRELACIONADO, SE BUSCARÍAn LAS DOS COLUMNAS CON MAYOR
        correlation_mat = df.corr()  # CORRELACIÓN ENTRE ELLAS. El proceso sería muy similar más adelante, y se explica después de este "if"
        corr_pairs = correlation_mat.unstack()
        lista = []
        lista = change_score(
            corr_pairs)  # Aplico el método "change_score" para crear una lista con todos los scores positivos y descartar los de ALTA CORRELACIÓN
        mayor = max(
            lista)  # A diferencia de más adelante, aquí se busca el mayor grado de correlación entre dos columnas (sin target)
        indice = lista.index(mayor)  # Ubicar índice de la mayor correlación entre columnas
        x = corr_pairs.index[
            indice]  # Localizar el Multiíndice que corresponde con el mayor grado de correlación
        lista_final.append(df.columns[-1])  # Añadir nombre del Target
        lista_final.append(x[0])  # Añadir nombre de la priemra columna
        lista_final.append(x[1])  # Añadir nombre de la segunda coolumna
        return lista_final

    corr_pairs = corr_pairs[(lista[0]):(lista[
                                            -1] + 1)]  # Reducir el Series Multiíndice solo a los elementos que correspondan a las correlaciones con el target

    lista_2 = []
    lista_2 = change_score(corr_pairs)

    lista_3 = lista_2.copy()
    lista_3.sort(reverse=True)  # Ordenar de mayor a menor

    if n_columns >= 4:  # El funcionamiento de esta es igual para todos los "if"
        mayor_4 = lista_3[3]  # Obtener el cuarto valor de correlación máximo
        indice_4 = lista_2.index(
            mayor_4)  # Obtener el índice del cuarto valor, con respecto al series Multiíndice
        d = corr_pairs.index[
            indice_4]  # Obtener el el índice del Multiíndice (el cual incluye el nombre del target y la columna correspondiente)
        lista_final.append(d[1])  # Añadir nombre de la cuarta columna con mayor correlacón con el targer

    if n_columns >= 3:
        mayor_3 = lista_3[2]
        indice_3 = lista_2.index(mayor_3)
        c = corr_pairs.index[indice_3]
        lista_final.append(c[1])  # Añadir nombre de la tercera columna con mayor correlacón con el targer

    if n_columns >= 2:
        mayor_2 = lista_3[1]
        indice_2 = lista_2.index(mayor_2)
        b = corr_pairs.index[indice_2]
        lista_final.append(b[1])  # Añadir nombre de la segunda columna con mayor correlacón con el targer

    if n_columns >= 1:
        mayor_1 = lista_3[0]
        indice_1 = lista_2.index(mayor_1)
        a = corr_pairs.index[indice_1]
        lista_final.append(a[1])  # Añadir nombre de la columna con mayor correlacón con el targer
        lista_final.append(a[0])  # Añadir nombre del Target

    lista_final.reverse()

    return lista_final


def knn (X_train, X_test, y_train, y_test):

    """
    Function to implement a classification algorithm such as KNeighborsClassifier.

    :param X_train: X_train
    :param y_train: y_train
    :param X_test: X_test
    :param y_test: y_test
    :return: fitted model and metrics

    In case the Dataframe does not fit the classifier, returns None.
    """

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    try:
        if len (X_train.columns) < 2:
            models = None
            scores = []
            return models, scores

        if len (X_train.columns) > 2:
            corr = corr_target (X_train, y_train, 2)
            colum1 = corr[1]
            colum2 = corr[2]

            X_train = pd.concat([X_train[colum1], X_train[colum2]], axis=1)
            X_test = pd.concat([X_test[colum1], X_test[colum2]], axis=1)

        scores_train = []
        scores_test = []
        k_range = range(1,20)

        for i in k_range:

            knn_f = KNeighborsClassifier(n_neighbors= i)
            knn_f.fit(X_train, y_train)

            scores_train.append(knn_f.score(X_train, y_train))
            scores_test.append(knn_f.score(X_test, y_test))

        scores_change = change_score (scores_test)

        try:
            k = scores_test.index(max(scores_change)) + 1

        except:
            k = scores_test.index(min(scores_test)) + 1

        knn = KNeighborsClassifier(n_neighbors= k)

        model = knn.fit(X_train, y_train)

        metrics  =  model_scoring_classification ('KNN' , model , X_test , y_test,  average="weighted")

        return model, metrics

    except ValueError:

        print ("Dataframe does not fit classifier")

        return None


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


def lin_reg(X_train, X_test, y_train, y_test, regular_type=None):
    """
    Apply a lineal regression model with 3 polynomial levels
    :param X_train
    :param y_train
    :param X_test
    :param y_test
    :param regular_type: type of Regularization (ridge, lasso, elasticnet)
    :return model_fit, metrics: trained model and metrics
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
