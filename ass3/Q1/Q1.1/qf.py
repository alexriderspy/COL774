import xgboost as xgb
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

train_path = sys.argv[1]
val_path = sys.argv[2]
test_path = sys.argv[3]

train_data = pd.read_csv(train_path+'/train.csv')
val_data = pd.read_csv(val_path + '/val.csv')
test_data = pd.read_csv(test_path + '/test.csv')

train_data[train_data == '?'] = np.nan

val_data[val_data == '?'] = np.nan

test_data[test_data == '?'] = np.nan

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

arrX = (train_data)
arrX = np.delete(arrX,4,axis=1)

val_arrX = (val_data)
val_arrX = np.delete(val_arrX,4,axis=1)

test_arrX = (test_data)
test_arrX = np.delete(test_arrX,4,axis=1)

parameters = {'n_estimators': [10,20,30,40,50], 'subsample' : [0.1,0.2,0.3,0.4,0.5,0.6], 'max_depth' : [4,5,6,7,8,9,10]}

clf = xgb.XGBClassifier(objective="binary:logistic")

tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
tree_clf = tree_clf.fit(arrX,arrY)
tree_clf=tree_clf.best_estimator_
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