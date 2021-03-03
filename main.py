import pandas as pd
import numpy as np

# --sklearn--
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn import metrics

# --local files--
import dataset_utility
import preprocess

df = pd.read_csv(
    "mushrooms-filled-missing.csv", names=dataset_utility.get_column_names()
)

df_enc, df_enc_columns = preprocess.one_hot_encode(df)

df_enc = pd.DataFrame(df_enc, columns=df_enc_columns)

train, test = train_test_split(df_enc, test_size=0.3)

class_columns, feature_columns = dataset_utility.get_split_column_names(train)

train_features, train_labels = preprocess.split_features_labels(train, class_columns)
test_features, test_labels = preprocess.split_features_labels(test, class_columns)


# ---Decision Tree----
dt = DecisionTreeClassifier(random_state=0, max_depth=5)
dt.fit(train_features, train_labels)

predictions = dt.predict(test_features)

print("Accuracy:", metrics.accuracy_score(predictions, test_labels))