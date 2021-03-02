import pandas as pd
import numpy as np
import graphviz
import pydotplus

# from IPython.display import Image

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.metrics import accuracy_score


from matplotlib import pyplot as plt
import time

import preprocess

class_columns = [
    "stalk-root_b",
    "stalk-root_c",
    "stalk-root_e",
    "stalk-root_r",
    "stalk-root_?",
]


def get_column_names(dataset):

    feature_columns = dataset.columns.tolist()

    for class_column in class_columns:
        if class_column in feature_columns:
            feature_columns.remove(class_column)

    return class_columns, feature_columns


def insert_columns(dataset):

    dt = dataset.copy()

    for column in class_columns:
        dt.insert(df_enc.columns.get_loc(column), column, float("NaN"))

    print("length: " + str(len(dt.columns.tolist())))

    return dt


column_names = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

df = pd.read_csv("mushrooms.csv", names=column_names)

df_enc, df_enc_columns = preprocess.one_hot_encode(df)

# print(df_columns)

df_enc = pd.DataFrame(df_enc, columns=df_enc_columns)

# print(df)

train, missing = preprocess.extract_missing(df_enc)

train = pd.DataFrame(train, columns=df_enc_columns)
missing = pd.DataFrame(missing, columns=df_enc_columns)

train = train.reset_index(drop=True)
missing = missing.reset_index(drop=True)


class_columns, feature_columns = get_column_names(train)

missing = preprocess.remove_class_columns(missing, class_columns)

train_features, train_labels = preprocess.split_features_labels(train, class_columns)


dt = DecisionTreeClassifier(random_state=0, max_depth=5)
dt.fit(train_features, train_labels)

dt.score(train_features, train_labels)


predictions = dt.predict(missing)

fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(
    dt, feature_names=feature_columns, class_names=class_columns, filled=True
)
fig.savefig("decision_tree.png")

predictions = pd.DataFrame(predictions, columns=class_columns)

missing_combined = insert_columns(missing)

missing_combined = missing_combined.fillna(predictions)

# print(missing_combined)
# print(train)

full_combined = train.append(missing_combined)

full_combined = preprocess.one_hot_decode(full_combined)

full_combined = pd.DataFrame(full_combined, columns=column_names)

# print(full_combined.shape)
# print(df.shape)

full_combined.to_csv("mushrooms-filled-missing.csv", index=False, header=False)