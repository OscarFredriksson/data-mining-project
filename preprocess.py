import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

ohe = OneHotEncoder()
le = LabelEncoder()


def extract_missing(dataset):

    train = []
    missing = []

    for index, row in dataset.iterrows():

        if row["stalk-root_?"] == 1:
            missing.append(row)
        else:
            train.append(row)

    return train, missing


def remove_class_columns(dataset, class_names):
    ds = dataset.copy()

    for c in class_names:
        ds.pop(c)

    return ds


def split_features_labels(dataset, class_names):

    features = dataset.copy()

    features = remove_class_columns(features, class_names)

    labels = dataset[class_names].copy()

    return features, labels


def one_hot_encode(dataset):

    ohe.fit(dataset)

    return ohe.transform(dataset).toarray(), ohe.get_feature_names(dataset.columns)


# Will decode the latest onehotencoded dataset
def one_hot_decode(dataset):

    return ohe.inverse_transform(dataset)


def label_encode(dataset):
    # le.fit(dataset)

    return dataset.apply(le.fit_transform)


# Will decode the latest labelencoded dataset
def label_decode(dataset):
    return dataset.apply(le.inverse_transform)