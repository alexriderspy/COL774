from cgi import test
from sklearn.impute import SimpleImputer

import sys
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

train_path = sys.argv[1]
val_path = sys.argv[2]
test_path = sys.argv[3]

train_data_o = pd.read_csv(train_path+'/train.csv')
val_data_o = pd.read_csv(val_path + '/val.csv')
test_data_o = pd.read_csv(test_path + '/test.csv')

train_data_o[train_data_o == '?'] = np.nan

val_data_o[val_data_o == '?'] = np.nan

test_data_o[test_data_o == '?'] = np.nan


features = list(train_data_o.columns)

features.remove('Severity')
features.remove('BI-RADS assessment')

class_names = ['Benign','Malignant']

train_data = train_data_o
val_data = val_data_o
test_data = test_data_o

imp_mean = SimpleImputer(missing_values= np.nan, strategy='mean')
imp_mean.fit(train_data)

train_data = imp_mean.transform(train_data)
val_data = imp_mean.transform(val_data)
test_data = imp_mean.transform(test_data)

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

parameters = {'max_depth':[3], 'min_samples_split': [4,5,6], 'min_samples_leaf': [4,5,6]}

clf = tree.DecisionTreeClassifier()
tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
tree_clf = tree_clf.fit(arrX,arrY)
tree_clf=tree_clf.best_estimator_
print("imputer strategy is mean")
print(tree_clf)

tree_clf = tree_clf.fit(arrX,arrY)


ypred = tree_clf.predict(arrX)
train_acc = np.sum(ypred == arrY)/len(arrY)
print("Training accuracy : " + str(train_acc))

val_ypred = tree_clf.predict(val_arrX)
val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
print("Validation accuracy : " + str(val_acc))

test_ypred = tree_clf.predict(test_arrX)
test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
print("Test accuracy : " + str(test_acc))

train_data = train_data_o
val_data = val_data_o
test_data = test_data_o

imp_median = SimpleImputer(missing_values= np.nan, strategy='median')
imp_median.fit(train_data)

train_data = imp_median.transform(train_data)
val_data = imp_median.transform(val_data)
test_data = imp_median.transform(test_data)

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

parameters = {'max_depth':[3], 'min_samples_split': [4,5,6], 'min_samples_leaf': [4,5,6]}

clf = tree.DecisionTreeClassifier()
tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
tree_clf = tree_clf.fit(arrX,arrY)
tree_clf=tree_clf.best_estimator_
print("imputer strategy is median")
print(tree_clf)

tree_clf = tree_clf.fit(arrX,arrY)

ypred = tree_clf.predict(arrX)
train_acc = np.sum(ypred == arrY)/len(arrY)
print("Training accuracy : " + str(train_acc))

val_ypred = tree_clf.predict(val_arrX)
val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
print("Validation accuracy : " + str(val_acc))

test_ypred = tree_clf.predict(test_arrX)
test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
print("Test accuracy : " + str(test_acc))