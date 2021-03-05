import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
import pylab
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def read_mushroom_data():
    df=pd.read_csv('mushrooms.csv')
    return df

###################### Data preparation ####################
df = read_mushroom_data()

print('number of samples: ', df.shape[0])
print('number of attributes: ', df.shape[1])
print('\nValues classified as \'Missing\' for stalk-root: ', (df.iloc[:,11] == '?').sum())

### due to large number of missing values, this featues are removed
df = df.drop(str(df.columns[11]), axis = 1)

### alphabet to numerical 
df2 = pd.get_dummies(df)



print('\nNumber of samples: ', df2.shape[0])
print('Number of attributes: ', df2.shape[1])
print('\nRemaining missing values across all attributes and samples: ', df2.isnull().sum().sum())
print('\nMinimum value across all attributes and samples: ', df2.min().min())
print('Maximum value across all attributes and samples: ', df2.max().max())
print('\nMinimum fraction of \'1\'-s across all attributes: {:.5f}'.format(df2.mean().min()))
print('Maximum fraction of \'1\'-s across all attributes: {:.5f}'.format(df2.mean().max()))

# Define poisonous as 1 and edible as 0 for the target
X = df2.iloc[:,2:]
y = df2.iloc[:,1] 

#### Partition data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=None)

#### Reduce dimensionality of the features from 113 to two principal componets
#### A PCA plot converts the correlations (or lack there of) among all of the features into a 2-D vector
pca = PCA(n_components=2).fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

################### Analysis, Fitting and Visualizing Classifiers   ###################

### plot the target as a function of the two principal components
# plt.cla()
# plt.clf()

plt.figure(dpi=120)
plt.scatter(X_train[y_train.values==0, 0], X_train[y_train.values==0, 1], label = 'Edible', alpha = 0.5, s = 2)
plt.scatter(X_train[y_train.values==1, 0], X_train[y_train.values==1, 1], label = 'Poisonous', alpha = 0.5, s = 2)
plt.title('Mushroom Data Set\nFirst Two Principal Components')
plt.legend(frameon=1)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.gca().set_aspect('equal')
plt.savefig('knn.png')

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

  # Accuracy:
acc = accuracy_score(y_pred, y_test)
print("Accuracy:", acc)

confusion_matrix(y_pred,y_test)

k_fold_splits = 5

kf = KFold(n_splits = k_fold_splits, shuffle=true)
accuracies = []

# for i in range(f_fold_splits)