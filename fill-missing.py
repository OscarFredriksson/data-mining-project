import pandas as pd
import numpy as np

# import graphviz
# import pydotplus

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import tree
from sklearn.metrics import accuracy_score


from matplotlib import pyplot as plt
import time

# ---local files---
import preprocess
import dataset_utility


def get_class_column_names():
    class_columns = [
        "stalk-root_b",
        "stalk-root_c",
        "stalk-root_e",
        "stalk-root_r",
        "stalk-root_?",
    ]

    return class_columns


def insert_class_columns(dataset):

    dt = dataset.copy()

    for column in get_class_column_names():
        dt.insert(df_enc.columns.get_loc(column), column, float("NaN"))

    return dt


df = pd.read_csv("mushrooms.csv", names=dataset_utility.get_column_names())

df_enc, df_enc_columns = preprocess.one_hot_encode(df)

df_enc = pd.DataFrame(df_enc, columns=df_enc_columns)

train, missing = preprocess.extract_missing(df_enc)

train = pd.DataFrame(train, columns=df_enc_columns)
missing = pd.DataFrame(missing, columns=df_enc_columns)

train = train.reset_index(drop=True)
missing = missing.reset_index(drop=True)

class_columns, feature_columns = dataset_utility.get_split_column_names(
    train, get_class_column_names()
)

missing = preprocess.remove_class_columns(missing, class_columns)

train_features, train_labels = preprocess.split_features_labels(train, class_columns)


dt = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_leaf=10)
dt.fit(train_features, train_labels)

dt.score(train_features, train_labels)


predictions = dt.predict(missing)

# print(feature_columns)
# print(class_columns)

fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(
    dt,
    feature_names=feature_columns,
    class_names=class_columns,
    filled=True,
)
fig.savefig("decision_tree-missing-values.png")


predictions = pd.DataFrame(predictions, columns=get_class_column_names())

missing_combined = insert_class_columns(missing)

print(missing_combined["stalk-root_?"])

missing_combined = missing_combined.fillna(predictions)

# print(train)

full_combined = train.append(missing_combined)

full_combined = preprocess.one_hot_decode(full_combined)

full_combined = pd.DataFrame(full_combined, columns=dataset_utility.get_column_names())

full_combined.to_csv("mushrooms-filled-missing.csv", index=False, header=False)
