from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def divide(dataset, target_column):
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    return X, y


def prepare_data(dataset, target_column, sample_type=None, scaler_type='std', test_size=0.2):
    """
        Divides dataset in X and y, samples if necessary, divides in train and test and scales

        :param pandas dataset: Dataset we are working with
        :param str target_column: Target column name
        :param str sample: Sampling to be applied. Accepts under, SMOTE, both and none
        :param str scaler_type: Scaler to be applied. Accepts std, minmax and None
        :param float test_size: Test size
        :return list: tuple list with X and y and sampled X and y
    """
    sets = []
    splited_sets = []
    scaled_sets = []

    dataset = dataset.loc[:, dataset.dtypes != object]

    # 1. Divide in X and y
    X, y = divide(dataset, target_column)
    sets.append((X, y))

    # 2. Sample if necessary
    if sample_type == 'under':
        X_unders, y_unders = RandomUnderSampler(random_state=42).fit_resample(X, y)
        sets.append((X_unders, y_unders))

    elif sample_type == 'SMOTE':
        X_overs, y_overs = SMOTE(random_state=42).fit_resample(X, y)
        sets.append((X_overs, y_overs))

    elif sample_type == 'both':
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

    return scaled_sets
