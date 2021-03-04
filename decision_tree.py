import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# --local files--
import dataset_utility
import preprocess

# --sklearn--
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn import metrics


def get_class_column_names():
    class_columns = ["class_e", "class_p"]

    return class_columns


cm_labels = ["edible", "poisonous"]


def do_decision_tree(df):
    df_enc, df_enc_columns = preprocess.one_hot_encode(df)

    df_enc = pd.DataFrame(df_enc, columns=df_enc_columns)

    # ---KFold cross validation---

    k_fold_splits = 5

    kf = KFold(n_splits=k_fold_splits, shuffle=True)

    accuracies = []

    best_predictions = pd.DataFrame([], columns=get_class_column_names())

    dt = DecisionTreeClassifier(random_state=0, max_depth=4)

    for i in range(k_fold_splits):

        class_columns, feature_columns = dataset_utility.get_split_column_names(
            df, get_class_column_names()
        )

        features, labels = preprocess.split_features_labels(df_enc, class_columns)

        result = next(kf.split(df_enc), None)
        train_features = features.iloc[result[0]]
        test_features = features.iloc[result[1]]
        train_labels = labels.iloc[result[0]]
        test_labels = labels.iloc[result[1]]

        # ---Decision Tree----
        dt.fit(train_features, train_labels)

        predictions = dt.predict(test_features)

        accuracy = metrics.accuracy_score(predictions, test_labels)

        accuracies.append(accuracy)

        if len(accuracies) == 0 or accuracy >= max(accuracies):
            best_predictions = pd.DataFrame(
                predictions, columns=get_class_column_names()
            )

    predictions = pd.DataFrame(predictions, columns=get_class_column_names())

    predictions = predictions.idxmax(axis=1)
    test_labels = test_labels.idxmax(axis=1)

    df_cm = pd.DataFrame(
        metrics.confusion_matrix(
            predictions, test_labels, labels=get_class_column_names()
        ),
        index=cm_labels,
        columns=cm_labels,
    )

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt="g")

    plt.ylabel("Predicted")
    plt.xlabel("Actual")

    plt.savefig("decision-tree-cm.png")

    print("K-fold results: ", accuracies)
    print("Mean accuracy: ", np.mean(accuracies))
