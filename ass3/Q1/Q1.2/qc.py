import sys
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt

train_path = sys.argv[1]
val_path = sys.argv[2]
test_path = sys.argv[3]

train_data = pd.read_csv(train_path+'/DrugsComTrain.csv')

y_train = train_data.rating

train_data.drop('rating', inplace=True, axis=1)
train_data.drop('usefulCount', inplace=True, axis=1)

vectorizer1 = CountVectorizer()
X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
vectorizer2 = CountVectorizer()
X_train_review = vectorizer2.fit_transform(train_data.review)
vectorizer3 = CountVectorizer()
X_train_date = vectorizer3.fit_transform(train_data.date)

X_train = hstack([X_train_condition, X_train_review, X_train_date])

val_data = pd.read_csv(val_path + '/DrugsComVal.csv')

y_val = val_data.rating

val_data.drop('rating', inplace=True, axis=1)
val_data.drop('usefulCount', inplace=True, axis=1)

X_val_condition = vectorizer1.transform(val_data.condition.astype('U'))
X_val_review = vectorizer2.transform(val_data.review)
X_val_date = vectorizer3.transform(val_data.date)

X_val = hstack([X_val_condition, X_val_review, X_val_date])

clf = tree.DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
fig.savefig('Total Impurity vs effective alpha for training set.png')

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

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

train_scores = [clf.score(X_train, y_train) for clf in clfs]
val_scores = [clf.score(X_val, y_val) for clf in clfs]
#test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training, validation and test sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, val_scores, marker="o", label="validation", drawstyle="steps-post")
#ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
fig.savefig('accuracy vs alpha.png')


# dt.fit(X_train, y_train)

# y_pred = dt.predict(X_train)
# train_acc = np.sum(y_pred == y_train)/len(y_train)
# print("Training accuracy : " + str(train_acc))



# y_val_pred = dt.predict(X_val)
# val_acc = np.sum(y_val_pred == y_val)/len(y_val)
# print("Validation accuracy : " + str(val_acc))

# test_data = pd.read_csv(val_path + '/DrugsComTest.csv')

# y_test = test_data.rating

# test_data.drop('rating', inplace=True, axis=1)
# test_data.drop('usefulCount', inplace=True, axis=1)

# X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
# X_test_review = vectorizer2.transform(test_data.review)
# X_test_date = vectorizer3.transform(test_data.date)

# X_test = hstack([X_test_condition, X_test_review, X_test_date])

# y_test_pred = dt.predict(X_test)
# test_acc = np.sum(y_test_pred == y_test)/len(y_test)
# print("Test accuracy : " + str(test_acc))