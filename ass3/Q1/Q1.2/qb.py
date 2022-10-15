import sys
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

train_path = sys.argv[1]
val_path = sys.argv[2]
test_path = sys.argv[3]

train_data = pd.read_csv(train_path+'/train.csv')
val_data = pd.read_csv(val_path + '/val.csv')
test_data = pd.read_csv(test_path + '/test.csv')

train_data[train_data == '?'] = np.nan
train_data.dropna(inplace=True)
val_data[val_data == '?'] = np.nan
val_data.dropna(inplace=True)
test_data[test_data == '?'] = np.nan
test_data.dropna(inplace=True)

features = list(train_data.columns)

features.remove('Severity')
features.remove('BI-RADS assessment')

class_names = ['Benign','Malignant']
train_data = train_data.to_numpy()
val_data = val_data.to_numpy()
test_data = test_data.to_numpy()

train_data = np.delete(train_data,0,axis = 1)
val_data = np.delete(val_data,0,axis = 1)
test_data = np.delete(test_data,0,axis = 1)

def f(x):
    return int(x)

arrY = train_data[:,4].astype('int')

val_arrY = val_data[:,4].astype('int')
test_arrY = test_data[:,4].astype('int')

arrX = np.vectorize(f)(train_data)
arrX = np.delete(arrX,4,axis=1).astype('int')

val_arrX = np.vectorize(f)(val_data)
val_arrX = np.delete(val_arrX,4,axis=1).astype('int')

test_arrX = np.vectorize(f)(test_data)
test_arrX = np.delete(test_arrX,4,axis=1).astype('int')

parameters = {'max_depth':[3, 5, 8], 'min_samples_split': [4,5,6], 'min_samples_leaf': [4,5,6]}

clf = tree.DecisionTreeClassifier()
tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
tree_clf = tree_clf.fit(arrX,arrY)
tree_clf=tree_clf.best_estimator_
print(tree_clf)

tree_clf = tree_clf.fit(arrX,arrY)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=600)
tree.plot_tree(tree_clf,feature_names = features, class_names = class_names, filled=True, rounded = True)
fig.savefig('q1.1b.png')

ypred = tree_clf.predict(arrX)
train_acc = np.sum(ypred == arrY)/len(arrY)
print("Training accuracy : " + str(train_acc))

val_ypred = tree_clf.predict(val_arrX)
val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
print("Validation accuracy : " + str(val_acc))

test_ypred = tree_clf.predict(test_arrX)
test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
print("Test accuracy : " + str(test_acc))