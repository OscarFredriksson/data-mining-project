import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pylab

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
df2 = pd.get_dummies(df)

print('\nNumber of samples: ', df2.shape[0])
print('Number of attributes: ', df2.shape[1])
print('\nRemaining missing values across all attributes and samples: ', df2.isnull().sum().sum())
print('\nMinimum value across all attributes and samples: ', df2.min().min())
print('Maximum value across all attributes and samples: ', df2.max().max())
print('\nMinimum fraction of \'1\'-s across all attributes: {:.5f}'.format(df2.mean().min()))
print('Maximum fraction of \'1\'-s across all attributes: {:.5f}'.format(df2.mean().max()))

x=df2.iloc[:,df2.columns!='class']
y=df2.iloc[:,0]

print("X:\n", x)
print("Y:\n", y)
#### Partition data into training and test sets

#### Partition data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=None)

#### Reduce dimensionality of the features from 113 to two principal compinets
pca = PCA(n_components=2).fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

################### Analysis, Fitting and Visualizing Classifiers   ###################
plt.cla()
plt.clf()

plt.figure(dpi=120)
plt.scatter(X_train[y_train.values==0, 0], X_train[y_train.values==0, 1], label = 'Edible', alpha = 0.5, s = 2)
plt.scatter(X_train[y_train.values==1, 0], X_train[y_train.values==1, 1], label = 'Poisonous', alpha = 0.5, s = 2)
plt.title('Mushroom Data Set\nFirst Two Principal Components')
plt.legend(frameon=1)
plt.xlabel('PC 1')
plt.xlabel('PC 2')
plt.gca().set_aspect('equal')
plt.savefig('knn.png')


#######   Function to visualize the decision boundary and decision Probabilities  ##############
# def decision_boundary(X,y,fitted_model):
#     fig=plt.figure(figsize=(10,5), dpi=100)

#     for i, plot_type in enumerate(['Decision Boundary', 'Decision Probabilities']):
#         plt.subplot(1,2,i+1)
#         step = 0.01
#         x_max = X[_,0]