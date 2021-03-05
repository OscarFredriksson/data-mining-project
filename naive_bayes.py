import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

# local files
import preprocess
import dataset_utility

# sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn import metrics

class_column = ["class"]

cm_labels = ["edible", "poisonous"]


def do_naive_bayes(df):

    df_enc = preprocess.label_encode(df)

    # ---KFold cross validation---

    k_fold_splits = 5

    kf = KFold(n_splits=k_fold_splits, shuffle=True)

    accuracies = []

    best_predictions = pd.DataFrame([], columns=class_column)
    best_test_labels = pd.DataFrame([], columns=class_column)

    model = GaussianNB()

    for i in range(k_fold_splits):

        class_columns, feature_columns = dataset_utility.get_split_column_names(
            df, class_column
        )

        features, labels = preprocess.split_features_labels(df_enc, class_columns)

        result = next(kf.split(df_enc), None)
        train_features = features.iloc[result[0]]
        test_features = features.iloc[result[1]]
        train_labels = labels.iloc[result[0]]
        test_labels = labels.iloc[result[1]]

        # ---Naive Bayes---
        model.fit(train_features, train_labels.values.ravel())

        predictions = model.predict(test_features)

        accuracy = metrics.accuracy_score(predictions, test_labels)

        accuracies.append(accuracy)

        # print(accuracy > max(accuracies))
        # print(len(accuracies))

        if accuracy >= max(accuracies):
            best_predictions = pd.DataFrame(predictions, columns=class_column)
            best_test_labels = pd.DataFrame(test_labels, columns=class_column)

    print("K-fold results: ", accuracies)
    print("Mean accuracy: ", np.mean(accuracies))

    df_cm = pd.DataFrame(
        metrics.confusion_matrix(best_predictions, best_test_labels, labels=[0, 1]),
        index=cm_labels,
        columns=cm_labels,
    )

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt="g")

    plt.ylabel("Predicted")
    plt.xlabel("Actual")

    plt.savefig("naive-bayes-cm.png")
