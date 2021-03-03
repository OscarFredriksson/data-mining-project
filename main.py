import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# --sklearn--
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn import metrics

# --local files--
import dataset_utility
import preprocess


def get_class_column_names():
    class_columns = ["class_e", "class_p"]

    return class_columns


df = pd.read_csv(
    "mushrooms-filled-missing.csv", names=dataset_utility.get_column_names()
)

df_enc, df_enc_columns = preprocess.one_hot_encode(df)

df_enc = pd.DataFrame(df_enc, columns=df_enc_columns)

train, test = train_test_split(df_enc, test_size=0.3)

class_columns, feature_columns = dataset_utility.get_split_column_names(
    train, get_class_column_names()
)

train_features, train_labels = preprocess.split_features_labels(train, class_columns)
test_features, test_labels = preprocess.split_features_labels(test, class_columns)


# ---Decision Tree----
dt = DecisionTreeClassifier(random_state=0, max_depth=4)
dt.fit(train_features, train_labels)

predictions = dt.predict(test_features)

print("Accuracy:", metrics.accuracy_score(predictions, test_labels))

predictions = pd.DataFrame(predictions, columns=get_class_column_names())

predictions = predictions.idxmax(axis=1)
test_labels = test_labels.idxmax(axis=1)

# print(predictions)
# print(test_labels)


df_cm = pd.DataFrame(
    metrics.confusion_matrix(predictions, test_labels, labels=get_class_column_names()),
    index=get_class_column_names(),
    columns=get_class_column_names(),
)


plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt="g")

plt.ylabel("Predicted")
plt.xlabel("Actual")

plt.savefig("tree-confusion-matrix.png")