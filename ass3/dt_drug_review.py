import nltk
nltk.download('stopwords')

import sys
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from sklearn import ensemble
import matplotlib.pyplot as plt



train_path = sys.argv[1]
val_path = sys.argv[2]
test_path = sys.argv[3]
output_path = sys.argv[4]
q_part = sys.argv[5]

import warnings
warnings.filterwarnings("ignore")

out = ''

if q_part == 'a':

    train_data = pd.read_csv(train_path)

    y_train = train_data.rating

    train_data.drop('rating', inplace=True, axis=1)

    vectorizer1 = CountVectorizer(stop_words=stopwords.words())
    X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
    vectorizer2 = CountVectorizer(stop_words=stopwords.words())
    X_train_review = vectorizer2.fit_transform(train_data.review)
    vectorizer3 = CountVectorizer(stop_words=stopwords.words())
    X_train_date = vectorizer3.fit_transform(train_data.date)
    vectorizer4 = CountVectorizer(stop_words=stopwords.words())
    X_train_usefulCount = vectorizer4.fit_transform(train_data.usefulCount.astype('U'))

    X_train = hstack([X_train_condition, X_train_review, X_train_date,X_train_usefulCount])
    # X_test = vectorizer.transform(X_test)

    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_train)
    train_acc = np.sum(y_pred == y_train)/len(y_train)
    out += ("Training accuracy : " + str(train_acc)) + '\n'


    val_data = pd.read_csv(val_path)

    y_val = val_data.rating

    val_data.drop('rating', inplace=True, axis=1)

    X_val_condition = vectorizer1.transform(val_data.condition.astype('U'))
    X_val_review = vectorizer2.transform(val_data.review)
    X_val_date = vectorizer3.transform(val_data.date)
    X_val_usefulCount = vectorizer4.transform(val_data.usefulCount.astype('U'))

    X_val = hstack([X_val_condition, X_val_review, X_val_date,X_val_usefulCount])

    y_val_pred = dt.predict(X_val)
    val_acc = np.sum(y_val_pred == y_val)/len(y_val)
    out += ("Validation accuracy : " + str(val_acc)) + '\n'

    test_data = pd.read_csv(test_path)

    y_test = test_data.rating

    test_data.drop('rating', inplace=True, axis=1)

    X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
    X_test_review = vectorizer2.transform(test_data.review)
    X_test_date = vectorizer3.transform(test_data.date)
    X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

    X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

    y_test_pred = dt.predict(X_test)
    test_acc = np.sum(y_test_pred == y_test)/len(y_test)
    out += ("Test accuracy : " + str(test_acc)) + '\n'

    output_file = open(output_path + '/2_a.txt','w')
    output_file.write(out)

elif q_part == 'b':

    train_data = pd.read_csv(train_path)

    y_train = train_data.rating

    train_data.drop('rating', inplace=True, axis=1)

    vectorizer1 = CountVectorizer(stop_words=stopwords.words())
    X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
    vectorizer2 = CountVectorizer(stop_words=stopwords.words())
    X_train_review = vectorizer2.fit_transform(train_data.review)
    vectorizer3 = CountVectorizer(stop_words=stopwords.words())
    X_train_date = vectorizer3.fit_transform(train_data.date)
    vectorizer4 = CountVectorizer(stop_words=stopwords.words())
    X_train_usefulCount = vectorizer4.fit_transform(train_data.usefulCount.astype('U'))

    X_train = hstack([X_train_condition, X_train_review, X_train_date,X_train_usefulCount])
    # X_test = vectorizer.transform(X_test)
    parameters = {'max_depth':[40,100,200,500], 'min_samples_split':[1,2,3,4,5,6], 'min_samples_leaf':[1,2,3,4,5,6]}

    dt = tree.DecisionTreeClassifier()

    clf = GridSearchCV(estimator=dt, param_grid=parameters)
    clf = clf.fit(X_train,y_train)
    clf=clf.best_estimator_

    out += str(clf) + '\n'

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_train)
    train_acc = np.sum(y_pred == y_train)/len(y_train)
    out += ("Training accuracy : " + str(train_acc)) + '\n'


    val_data = pd.read_csv(val_path)

    y_val = val_data.rating

    val_data.drop('rating', inplace=True, axis=1)

    X_val_condition = vectorizer1.transform(val_data.condition.astype('U'))
    X_val_review = vectorizer2.transform(val_data.review)
    X_val_date = vectorizer3.transform(val_data.date)
    X_val_usefulCount = vectorizer4.transform(val_data.usefulCount.astype('U'))

    X_val = hstack([X_val_condition, X_val_review, X_val_date,X_val_usefulCount])

    y_val_pred = clf.predict(X_val)
    val_acc = np.sum(y_val_pred == y_val)/len(y_val)
    out += ("Validation accuracy : " + str(val_acc)) + '\n'

    test_data = pd.read_csv(test_path)
    y_test = test_data.rating

    test_data.drop('rating', inplace=True, axis=1)

    X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
    X_test_review = vectorizer2.transform(test_data.review)
    X_test_date = vectorizer3.transform(test_data.date)
    X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

    X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

    y_test_pred = clf.predict(X_test)
    test_acc = np.sum(y_test_pred == y_test)/len(y_test)
    out += ("Test accuracy : " + str(test_acc)) + '\n'

    output_file = open(output_path + '/2_b.txt','w')
    output_file.write(out)

elif q_part == 'c':

    train_data = pd.read_csv(train_path)

    y_train = train_data.rating

    train_data.drop('rating', inplace=True, axis=1)

    vectorizer1 = CountVectorizer(stop_words=stopwords.words())
    X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
    vectorizer2 = CountVectorizer(stop_words=stopwords.words())
    X_train_review = vectorizer2.fit_transform(train_data.review)
    vectorizer3 = CountVectorizer(stop_words=stopwords.words())
    X_train_date = vectorizer3.fit_transform(train_data.date)
    vectorizer4 = CountVectorizer(stop_words=stopwords.words())
    X_train_usefulCount = vectorizer4.fit_transform(train_data.usefulCount.astype('U'))

    X_train = hstack([X_train_condition, X_train_review, X_train_date,X_train_usefulCount])
    # X_test = vectorizer.transform(X_test)

    val_data = pd.read_csv(val_path)

    y_val = val_data.rating

    val_data.drop('rating', inplace=True, axis=1)

    X_val_condition = vectorizer1.transform(val_data.condition.astype('U'))
    X_val_review = vectorizer2.transform(val_data.review)
    X_val_date = vectorizer3.transform(val_data.date)
    X_val_usefulCount = vectorizer4.transform(val_data.usefulCount.astype('U'))

    X_val = hstack([X_val_condition, X_val_review, X_val_date,X_val_usefulCount])

    test_data = pd.read_csv(test_path)
    y_test = test_data.rating

    test_data.drop('rating', inplace=True, axis=1)

    X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
    X_test_review = vectorizer2.transform(test_data.review)
    X_test_date = vectorizer3.transform(test_data.date)
    X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

    X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

    parameters = {'max_depth':[40,100,200,500], 'min_samples_split':[1,2,3,4,5,6], 'min_samples_leaf':[1,2,3,4,5,6]}

    clf = tree.DecisionTreeClassifier(random_state = 0)
    path = clf.cost_complexity_pruning_path(X_train,y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    fig.savefig(output_path + '/Total Impurity vs effective alpha for training set.png')

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
    fig.savefig(output_path + '/Depth vs alpha.png')

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    val_scores = [clf.score(X_val, y_val) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training, validation and test sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, val_scores, marker="o", label="validation", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    fig.savefig(output_path + '/accuracy vs alpha.png')


    #best tree is one with highest validation accuracy 
    #ccp_alpha = 0.02

    clf = tree.DecisionTreeClassifier(ccp_alpha = 0.02)
    out += str(clf) + '\n'
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_train)
    train_acc = np.sum(y_pred == y_train)/len(y_train)
    out += ("Training accuracy : " + str(train_acc)) + '\n'

    y_val_pred = clf.predict(X_val)
    val_acc = np.sum(y_val_pred == y_val)/len(y_val)
    out += ("Validation accuracy : " + str(val_acc)) + '\n'

    y_test_pred = clf.predict(X_test)
    test_acc = np.sum(y_test_pred == y_test)/len(y_test)
    out += ("Test accuracy : " + str(test_acc)) + '\n'

    output_file = open(output_path + '/2_c.txt','w')
    output_file.write(out)

elif q_part == 'd':

    train_data = pd.read_csv(train_path)

    y_train = train_data.rating

    train_data.drop('rating', inplace=True, axis=1)

    vectorizer1 = CountVectorizer(stop_words=stopwords.words())
    X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
    vectorizer2 = CountVectorizer(stop_words=stopwords.words())
    X_train_review = vectorizer2.fit_transform(train_data.review)
    vectorizer3 = CountVectorizer(stop_words=stopwords.words())
    X_train_date = vectorizer3.fit_transform(train_data.date)
    vectorizer4 = CountVectorizer(stop_words=stopwords.words())
    X_train_usefulCount = vectorizer4.fit_transform(train_data.usefulCount.astype('U'))

    X_train = hstack([X_train_condition, X_train_review, X_train_date,X_train_usefulCount])
    # X_test = vectorizer.transform(X_test)
    parameters = {'max_features': np.arange(0.4,1.0,0.1), 'min_samples_split': np.arange(2,10,2), 'n_estimators': np.arange(50,460,50), 'oob_score': [True]}

    dt = ensemble.RandomForestClassifier()

    clf = GridSearchCV(estimator=dt, param_grid=parameters)
    clf = clf.fit(X_train,y_train)
    clf=clf.best_estimator_

    out += str(clf) + '\n'

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_train)
    train_acc = np.sum(y_pred == y_train)/len(y_train)
    out += ("Training accuracy : " + str(train_acc)) + '\n'

    out += "Out of bag accuracy : " + str(clf.oob_score_) + '\n'

    val_data = pd.read_csv(val_path)

    y_val = val_data.rating

    val_data.drop('rating', inplace=True, axis=1)

    X_val_condition = vectorizer1.transform(val_data.condition.astype('U'))
    X_val_review = vectorizer2.transform(val_data.review)
    X_val_date = vectorizer3.transform(val_data.date)
    X_val_usefulCount = vectorizer4.transform(val_data.usefulCount.astype('U'))

    X_val = hstack([X_val_condition, X_val_review, X_val_date,X_val_usefulCount])

    y_val_pred = clf.predict(X_val)
    val_acc = np.sum(y_val_pred == y_val)/len(y_val)
    out += ("Validation accuracy : " + str(val_acc)) + '\n'

    test_data = pd.read_csv(test_path)
    y_test = test_data.rating

    test_data.drop('rating', inplace=True, axis=1)

    X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
    X_test_review = vectorizer2.transform(test_data.review)
    X_test_date = vectorizer3.transform(test_data.date)
    X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

    X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

    y_test_pred = clf.predict(X_test)
    test_acc = np.sum(y_test_pred == y_test)/len(y_test)
    out += ("Test accuracy : " + str(test_acc)) + '\n'

    output_file = open(output_path + '/2_d.txt','w')
    output_file.write(out)

elif q_part == 'e':

    train_data = pd.read_csv(train_path)

    y_train = train_data.rating

    train_data.drop('rating', inplace=True, axis=1)

    vectorizer1 = CountVectorizer(stop_words=stopwords.words())
    X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
    vectorizer2 = CountVectorizer(stop_words=stopwords.words())
    X_train_review = vectorizer2.fit_transform(train_data.review)
    vectorizer3 = CountVectorizer(stop_words=stopwords.words())
    X_train_date = vectorizer3.fit_transform(train_data.date)
    vectorizer4 = CountVectorizer(stop_words=stopwords.words())
    X_train_usefulCount = vectorizer4.fit_transform(train_data.usefulCount.astype('U'))

    X_train = hstack([X_train_condition, X_train_review, X_train_date,X_train_usefulCount])
    # X_test = vectorizer.transform(X_test)
    parameters = {'max_depth': np.arange(40,70,10), 'subsample': np.arange(0.4,0.8,0.1), 'n_estimators': np.arange(50,450,50)}

    dt = xgb.XGBClassifier(objective="binary:logistic")

    y_train -= 1
    
    clf = GridSearchCV(estimator=dt, param_grid=parameters)
    clf = clf.fit(X_train,y_train)
    clf=clf.best_estimator_

    out += str(clf) + '\n'

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_train)
    train_acc = np.sum(y_pred == y_train)/len(y_train)
    out += ("Training accuracy : " + str(train_acc)) + '\n'


    val_data = pd.read_csv(val_path)

    y_val = val_data.rating
    y_val -= 1
    val_data.drop('rating', inplace=True, axis=1)

    X_val_condition = vectorizer1.transform(val_data.condition.astype('U'))
    X_val_review = vectorizer2.transform(val_data.review)
    X_val_date = vectorizer3.transform(val_data.date)
    X_val_usefulCount = vectorizer4.transform(val_data.usefulCount.astype('U'))

    X_val = hstack([X_val_condition, X_val_review, X_val_date,X_val_usefulCount])

    y_val_pred = clf.predict(X_val)
    val_acc = np.sum(y_val_pred == y_val)/len(y_val)
    out += ("Validation accuracy : " + str(val_acc)) + '\n'

    test_data = pd.read_csv(test_path)
    y_test = test_data.rating
    y_test -= 1
    test_data.drop('rating', inplace=True, axis=1)

    X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
    X_test_review = vectorizer2.transform(test_data.review)
    X_test_date = vectorizer3.transform(test_data.date)
    X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

    X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

    y_test_pred = clf.predict(X_test)
    test_acc = np.sum(y_test_pred == y_test)/len(y_test)
    out += ("Test accuracy : " + str(test_acc)) + '\n'

    output_file = open(output_path + '/2_e.txt','w')
    output_file.write(out)

elif q_part == 'f':

    train_data = pd.read_csv(train_path)

    y_train = train_data.rating
    y_train -= 1.0
    train_data.drop('rating', inplace=True, axis=1)

    vectorizer1 = CountVectorizer(dtype = np.float64)
    X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
    vectorizer2 = CountVectorizer(dtype = np.float64)
    X_train_review = vectorizer2.fit_transform(train_data.review)
    vectorizer3 = CountVectorizer(dtype = np.float64)
    X_train_date = vectorizer3.fit_transform(train_data.date)
    vectorizer4 = CountVectorizer(dtype = np.float64)
    X_train_usefulCount = vectorizer4.fit_transform(train_data.usefulCount.astype('U'))

    X_train = hstack([X_train_condition, X_train_review, X_train_date,X_train_usefulCount])
    
    parameters = {'max_depth': np.arange(40,500,10), 'subsample': np.arange(0.4,2.0,0.1), 'n_estimators': np.arange(50,2000,50)}

    dt = lgb.LGBMClassifier()
    clf = GridSearchCV(dt, param_grid = parameters)
    clf.fit(X_train,y_train)
    out += str(clf) + '\n'
    y_pred = clf.predict(X_train)
    train_acc = np.sum(y_pred == y_train)/len(y_train)
    out += ("Training accuracy : " + str(train_acc)) + '\n'


    val_data = pd.read_csv(val_path)

    y_val = val_data.rating
    y_val -= 1.0
    val_data.drop('rating', inplace=True, axis=1)

    X_val_condition = vectorizer1.transform(val_data.condition.astype('U'))
    X_val_review = vectorizer2.transform(val_data.review)
    X_val_date = vectorizer3.transform(val_data.date)
    X_val_usefulCount = vectorizer4.transform(val_data.usefulCount.astype('U'))

    X_val = hstack([X_val_condition, X_val_review, X_val_date,X_val_usefulCount])

    y_val_pred = clf.predict(X_val)
    val_acc = np.sum(y_val_pred == y_val)/len(y_val)
    out += ("Validation accuracy : " + str(val_acc)) + '\n'

    test_data = pd.read_csv(test_path)
    y_test = test_data.rating
    y_test -= 1.0
    test_data.drop('rating', inplace=True, axis=1)

    X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
    X_test_review = vectorizer2.transform(test_data.review)
    X_test_date = vectorizer3.transform(test_data.date)
    X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

    X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

    y_test_pred = clf.predict(X_test)
    test_acc = np.sum(y_test_pred == y_test)/len(y_test)
    out += ("Test accuracy : " + str(test_acc)) + '\n'

    output_file = open(output_path + '/2_f.txt','w')
    output_file.write(out)

elif q_part == 'g':

    train_data_o = pd.read_csv(train_path)

    
    for num_samples in [20000,40000,60000,80000,100000,120000,140000,160000]:
        train_accuracies_grid = []
        test_accuracies_grid = []

        train_accuracies_ccp = []
        test_accuracies_ccp = []

        train_accuracies_rf = []
        test_accuracies_rf = []

        train_accuracies_xgb = []
        test_accuracies_xgb = []

        train_accuracies_lgb = []
        test_accuracies_lgb = []

        out += ("Number of samples: " + str(num_samples)) + '\n'
        train_data = train_data_o.sample(replace=True, random_state=1, n=num_samples)

        y_train = train_data.rating

        train_data.drop('rating', inplace=True, axis=1)

        vectorizer1 = CountVectorizer(stop_words=stopwords.words())
        X_train_condition = vectorizer1.fit_transform(train_data.condition.astype('U'))
        vectorizer2 = CountVectorizer(stop_words=stopwords.words())
        X_train_review = vectorizer2.fit_transform(train_data.review)
        vectorizer3 = CountVectorizer(stop_words=stopwords.words())
        X_train_date = vectorizer3.fit_transform(train_data.date)
        vectorizer4 = CountVectorizer(stop_words=stopwords.words())
        X_train_usefulCount = vectorizer4.fit_transform(train_data.usefulCount.astype('U'))

        out += ("Gridsearch") + '\n'
        X_train = hstack([X_train_condition, X_train_review, X_train_date,X_train_usefulCount])
        # X_test = vectorizer.transform(X_test)
        parameters = {'max_depth':[40,100,200,500], 'min_samples_split':[1,2,3,4,5,6], 'min_samples_leaf':[1,2,3,4,5,6]}

        dt = tree.DecisionTreeClassifier()

        clf = GridSearchCV(estimator=dt, param_grid=parameters)
        clf = clf.fit(X_train,y_train)
        clf=clf.best_estimator_

        out += str(clf) + '\n'

        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_train)
        train_acc = np.sum(y_pred == y_train)/len(y_train)
        train_accuracies_grid.append(train_acc)

        test_data = pd.read_csv(test_path)
        y_test = test_data.rating

        test_data.drop('rating', inplace=True, axis=1)

        X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
        X_test_review = vectorizer2.transform(test_data.review)
        X_test_date = vectorizer3.transform(test_data.date)
        X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

        X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

        y_test_pred = clf.predict(X_test)
        test_acc = np.sum(y_test_pred == y_test)/len(y_test)
        test_accuracies_grid.append(test_acc)

        out += ("CCP Alphas") + '\n'
        parameters = {'max_depth':[40,100,200,500], 'min_samples_split':[1,2,3,4,5,6], 'min_samples_leaf':[1,2,3,4,5,6]}

        clf = tree.DecisionTreeClassifier(random_state = 0)
        path = clf.cost_complexity_pruning_path(X_train,y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)

        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]

        train_scores = [clf.score(X_train, y_train) for clf in clfs]
        val_scores = [clf.score(X_val, y_val) for clf in clfs]
        test_scores = [clf.score(X_test, y_test) for clf in clfs]



        #best tree is one with highest validation accuracy 
        #ccp_alpha = 0.02

        clf = tree.DecisionTreeClassifier(ccp_alpha = 0.02)
        out += str(clf) + '\n'
        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_train)
        train_acc = np.sum(y_pred == y_train)/len(y_train)
        train_accuracies_ccp.append(train_acc)

        test_data = pd.read_csv(test_path)
        y_test = test_data.rating

        test_data.drop('rating', inplace=True, axis=1)

        X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
        X_test_review = vectorizer2.transform(test_data.review)
        X_test_date = vectorizer3.transform(test_data.date)
        X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

        X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

        y_test_pred = clf.predict(X_test)
        test_acc = np.sum(y_test_pred == y_test)/len(y_test)
        test_accuracies_ccp.append(test_acc)

        out += ("Random Forest") + '\n'
        parameters = {'max_features': np.arange(0.4,1.0,0.1), 'min_samples_split': np.arange(2,10,2), 'n_estimators': np.arange(50,460,50), 'oob_score': [True]}

        dt = ensemble.RandomForestClassifier()

        clf = GridSearchCV(estimator=dt, param_grid=parameters)
        clf = clf.fit(X_train,y_train)
        clf=clf.best_estimator_

        out += str(clf)  + '\n'

        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_train)
        train_acc = np.sum(y_pred == y_train)/len(y_train)
        train_accuracies_rf.append(train_acc)

        test_data = pd.read_csv(test_path)
        y_test = test_data.rating

        test_data.drop('rating', inplace=True, axis=1)

        X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
        X_test_review = vectorizer2.transform(test_data.review)
        X_test_date = vectorizer3.transform(test_data.date)
        X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

        X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

        y_test_pred = clf.predict(X_test)
        test_acc = np.sum(y_test_pred == y_test)/len(y_test)
        test_accuracies_rf.append(test_acc)

        out += ("XGBoost") + '\n'
        parameters = {'max_depth': np.arange(40,70,10), 'subsample': np.arange(0.4,0.8,0.1), 'n_estimators': np.arange(50,450,50), 'oob_score': [True]}

        dt = xgb.XGBClassifier(objective="binary:logistic")

        y_train -= 1
        
        clf = GridSearchCV(estimator=dt, param_grid=parameters)
        clf = clf.fit(X_train,y_train)
        clf=clf.best_estimator_

        out += str(clf)  + '\n'

        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_train)
        train_acc = np.sum(y_pred == y_train)/len(y_train)
        train_accuracies_xgb.append(train_acc)


        test_data = pd.read_csv(test_path)
        y_test = test_data.rating
        y_test -= 1
        test_data.drop('rating', inplace=True, axis=1)

        X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
        X_test_review = vectorizer2.transform(test_data.review)
        X_test_date = vectorizer3.transform(test_data.date)
        X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

        X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

        y_test_pred = clf.predict(X_test)
        test_acc = np.sum(y_test_pred == y_test)/len(y_test)
        test_accuracies_xgb.append(test_acc)

        out += ("LightGBM") + '\n'

        parameters = {'max_depth':[40,100,200,500], 'min_samples_split':[1,2,3,4,5,6], 'min_samples_leaf':[1,2,3,4,5,6]}

        clf = lgb.LGBMClassifier()
        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_train)
        train_acc = np.sum(y_pred == y_train)/len(y_train)
        train_accuracies_lgb.append(train_acc)

        test_data = pd.read_csv(test_path)
        y_test = test_data.rating
        y_test -= 1
        test_data.drop('rating', inplace=True, axis=1)

        X_test_condition = vectorizer1.transform(test_data.condition.astype('U'))
        X_test_review = vectorizer2.transform(test_data.review)
        X_test_date = vectorizer3.transform(test_data.date)
        X_test_usefulCount = vectorizer4.transform(test_data.usefulCount.astype('U'))

        X_test = hstack([X_test_condition, X_test_review, X_test_date,X_test_usefulCount])

        y_test_pred = clf.predict(X_test)
        test_acc = np.sum(y_test_pred == y_test)/len(y_test)
        test_accuracies_lgb.append(test_acc)

    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_grid)
    plt.savefig(output_path + '/grid_train.png')
    
    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_ccp)
    plt.savefig(output_path + '/ccp_train.png')
    
    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_rf)
    plt.savefig(output_path + '/rf_train.png')
    
    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_xgb)
    plt.savefig(output_path + '/xgb_train.png')

    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],train_accuracies_lgb)
    plt.savefig(output_path + '/lgb_train.png')
    
    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_grid)
    plt.savefig(output_path + '/grid_test.png')
    
    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_ccp)
    plt.savefig(output_path + '/ccp_test.png')
    
    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_rf)
    plt.savefig(output_path + '/rf_test.png')
    
    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_xgb)
    plt.savefig(output_path + '/xgb_test.png')

    plt.figure()
    plt.plot([20000,40000,60000,80000,100000,120000,140000,160000],test_accuracies_lgb)
    plt.savefig(output_path + '/lgb_test.png')

    output_file = open(output_path + '/2_g.txt','w')
    output_file.write(out)


