import sys
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt


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

train_data = train_data.to_numpy()
val_data = val_data.to_numpy()
test_data = test_data.to_numpy()

train_data = np.delete(train_data,0,axis = 1)
val_data = np.delete(val_data,0,axis = 1)
test_data = np.delete(test_data,0,axis = 1)

def f(x):
    return int(x)

arrY = train_data[:,4].astype('int')
arrX = np.vectorize(f)(train_data)
arrX = np.delete(arrX,4,axis=1).astype('int')
print(arrX)
print(arrY)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(arrX,arrY)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=1200)
tree.plot_tree(clf)
fig.savefig('q1.1a.png')
