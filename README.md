%%time

import sys
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import lightgbm as lgb

train_path = '.'
val_path = '.'
test_path = '.'

train_data = pd.read_csv(train_path+'/DrugsComTrain.csv')

y_train = train_data.rating
y_train-=1
train_data.drop('rating', inplace=True, axis=1)

vectorizer1 = CountVectorizer(dtype = np.float32)
X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
vectorizer2 = CountVectorizer(dtype = np.float32)
X_train_review = vectorizer2.fit_transform(train_data.review)
vectorizer3 = CountVectorizer(dtype = np.float32)
X_train_date = vectorizer3.fit_transform(train_data.date)
vectorizer4 = CountVectorizer(dtype = np.float32)
X_train_usefulCount = vectorizer4.fit_transform(train_data.usefulCount.astype('U'))

X_train = hstack([X_train_condition, X_train_review, X_train_date,X_train_usefulCount])
# X_test = vectorizer.transform(X_test)

dt_train = lgb.Dataset(X_train, label=y_train)

params = {'objective':'multiclass', 'max_depth':180, 'num_class':10, 'num_leaves':1000}

clf = lgb.train(params, dt_train)

y_pred = clf.predict(X_train)
train_acc = np.sum(y_pred == y_train)/len(y_train)
print("Training accuracy : " + str(train_acc))


val_data = pd.read_csv(val_path + '/DrugsComVal.csv')

y_val = val_data.rating

val_data.drop('rating', inplace=True, axis=1)

X_val_condition = vectorizer1.transform(val_data.condition.astype('U'))
X_val_review = vectorizer2.transform(val_data.review)
X_val_date = vectorizer3.transform(val_data.date)
X_val_usefulCount = vectorizer4.transform(val_data.usefulCount.astype('U'))

X_val = hstack([X_val_condition, X_val_review, X_val_date,X_val_usefulCount])

y_val_pred = clf.predict(X_val)
val_acc = np.sum(y_val_pred == y_val)/len(y_val)
print("Validation accuracy : " + str(val_acc))

test_data = pd.read_csv(test_path + '/DrugsComTest.csv')

y_test = test_data.rating

test_data.drop('rating', inplace=True, axis=1)

X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
X_test_review = vectorizer2.transform(test_data.review)
X_test_date = vectorizer3.transform(test_data.date)
X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

y_test_pred = clf.predict(X_test)
test_acc = np.sum(y_test_pred == y_test)/len(y_test)
print("Test accuracy : " + str(test_acc))
