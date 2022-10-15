import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sys
import numpy as np
import pandas as pd

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


clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(arrX, arrY)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
fig.savefig('Total Impurity vs effective alpha for training set.png')

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(arrX, arrY)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
fig.savefig('Depth vs alpha.png')

train_scores = [clf.score(arrX, arrY) for clf in clfs]
val_scores = [clf.score(val_arrX, val_arrY) for clf in clfs]
test_scores = [clf.score(test_arrX, test_arrY) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, val_scores, marker="o", label="validation", drawstyle="steps-post")
ax.legend()
fig.savefig('accuracy vs alpha.png')