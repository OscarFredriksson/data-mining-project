import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
import pylab
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import preprocess
import dataset_utility


def get_class_column_names():
    class_columns = ["class_e", "class_p"]

    return class_columns


def read_mushroom_data():
    df = pd.read_csv('mushrooms-filled-missing.csv')
    return df


###################### Data preparation ####################
df = read_mushroom_data()

print('number of samples: ', df.shape[0])
print('number of attributes: ', df.shape[1])
print('\nValues classified as \'Missing\' for stalk-root: ',
      (df.iloc[:, 11] == '?').sum())

df2, df2_columns = preprocess.one_hot_encode(df)
df2 = pd.DataFrame(df2, columns=df2_columns)

print('\nNumber of samples: ', df2.shape[0])
print('Number of attributes: ', df2.shape[1])
print('\nRemaining missing values across all attributes and samples: ',
      df2.isnull().sum().sum())
print('\nMinimum value across all attributes and samples: ', df2.min().min())
print('Maximum value across all attributes and samples: ', df2.max().max())
print(
    '\nMinimum fraction of \'1\'-s across all attributes: {:.5f}'.format(df2.mean().min()))
print(
    'Maximum fraction of \'1\'-s across all attributes: {:.5f}'.format(df2.mean().max()))

kf = KFold(n_splits=5, shuffle=True)
knn = KNeighborsClassifier(n_neighbors=10)
accuracies = []
best_predictions = pd.DataFrame([], columns=df2_columns)
best_test_labels = pd.DataFrame([], columns=df2_columns)

class_columns, feature_columns = dataset_utility.get_split_column_names(
    df2, get_class_column_names())

for i in range(2):
    result = next(kf.split(df2), None)
    # Define poisonous as 1 and edible as 0 for the target
    x = df2.iloc[:, 2:]
    y = df2.iloc[:, 1]

    x_train = x.iloc[result[0]]
    x_test = x.iloc[result[1]]
    y_train = y.iloc[result[0]]
    y_test = y.iloc[result[1]]

    pca = PCA(n_components=2).fit(x_train)

    # Reduce dimensionality of the features from 113 to two principal componets
    # A PCA plot converts the correlations (or lack there of) among all of the features into a 2-D vector
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    plt.figure(dpi=120)
    plt.scatter(x_train[y_train.values == 0, 0],
                x_train[y_train.values == 0, 1], label='Edible', alpha=0.5, s=2)
    plt.scatter(x_train[y_train.values == 1, 0],
                x_train[y_train.values == 1, 1], label='Poisonous', alpha=0.5, s=2)
    plt.title('Mushroom Data Set\nFirst Two Principal Components')
    plt.legend(frameon=1)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.gca().set_aspect('equal')
    plt.savefig('knn.png')

    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    print("y_pred: ", y_pred)

    acc = accuracy_score(y_pred, y_test)
    print("Accuracy:", acc)

    accuracies.append(acc)

    if acc >= max(accuracies):
        best_predictions = y_pred
        best_test_labels = y_test


print("K-fold results: ", accuracies)
print("Mean accuracy: ", np.mean(accuracies))
#print("Best prediction: ", best_prediction)

print(metrics.confusion_matrix(best_predictions, best_test_labels))

cm_labels = ["edible", "poisonous"]

df_cm = pd.DataFrame(
    metrics.confusion_matrix(
        best_predictions, best_test_labels, labels=[0.0, 1.0]
    ),
    index=cm_labels,
    columns=cm_labels,
)

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt="g")

plt.ylabel("Predicted")
plt.xlabel("Actual")

plt.savefig("knn-cm.png")
