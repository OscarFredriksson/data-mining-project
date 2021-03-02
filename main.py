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
        dt.insert(df.columns.get_loc(column), column, float("NaN"))

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

df, df_columns = preprocess.one_hot_encode(df)

# print(df_columns)

df = pd.DataFrame(df, columns=df_columns)

# print(df)

train, missing = preprocess.extract_missing(df)

train = pd.DataFrame(train, columns=df.columns.tolist())
missing = pd.DataFrame(missing, columns=df.columns.tolist())

train = train.reset_index(drop=True)
missing = missing.reset_index(drop=True)
# missing_features = preprocess.one_hot_encode(missing)

# train = preprocess.one_hot_encode(train)

# print("TRAIN:")
# print(train.columns)

# print("\nMiSSING:")
# print(missing["stalk-root"])


class_columns, feature_columns = get_column_names(train)

missing = preprocess.remove_class_columns(missing, class_columns)

train_features, train_labels = preprocess.split_features_labels(train, class_columns)

# train, test = train_test_split(train, test_size=0.0)

# train_features, train_labels = preprocess.split_features_labels(train, class_columns)
# test_features, test_labels = preprocess.split_features_labels(test, class_columns)

# print(train_features)
# print(train_labels)

dt = DecisionTreeClassifier(random_state=0, max_depth=5)
dt.fit(train_features, train_labels)

dt.score(train_features, train_labels)

# predictions = dt.predict(test_features)
# print(accuracy_score(test_labels, predictions))

predictions = dt.predict(missing)

predictions = pd.DataFrame(predictions, columns=class_columns)

# print(df.columns.get_loc("stalk-root_?"))

missing_combined = insert_columns(missing)

# print(missing.columns.tolist())

missing_combined = missing_combined.fillna(predictions)

missing_combined = preprocess.one_hot_decode(missing_combined)

missing_combined = pd.DataFrame(missing_combined, columns=column_names)

print(missing_combined)

missing_combined.to_csv("mushrooms-filled-missing.csv", index=False, header=False)

# train_combined = pd.concat([train_features, train_labels], axis=1)

# print(train_combined)

# full_combined = train_combined.append(missing_combined)

# full_combined = preprocess.one_hot_decode(full_combined)

# print(full_combined)

# train = preprocess.one_hot_decode(train)

# train = pd.DataFrame(train, columns=column_names)


# print(train_combined)

# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(
#     dt, feature_names=feature_columns, class_names=class_columns, filled=True
# )
# fig.savefig("decision_tree.png")

# completed_missing = pd.concat([missing, pd.DataFrame(predictions)], ignore_index=True)

# print(missing.columns.tolist())


# for index, row in missing.iterrows():
#     # for column in class_columns:
#     print(row[column])

# print(completed_missing)

# dot_data = export_graphviz(
#     dt,
#     out_file=None,
#     filled=True,
#     rounded=True,
#     # feature_names=feature_columns,
#     class_names=class_columns,
# )

# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
