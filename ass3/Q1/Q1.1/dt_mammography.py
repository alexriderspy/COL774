import sys
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn import ensemble
import xgboost as xgb


train_path = sys.argv[1]
val_path = sys.argv[2]
test_path = sys.argv[3]
output_path = sys.argv[4]
q_part = sys.argv[5]

train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)

warnings.simplefilter(action='ignore', category=FutureWarning)

out = ''
if q_part == 'a':
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

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(arrX,arrY)

    # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=600)
    # tree.plot_tree(clf,feature_names = features, class_names = class_names, filled=True, rounded = True)
    # fig.savefig('q1.1a.png')

    ypred = clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    output_file = open(output_path + '/1_a.txt','w')
    output_file.write(out)

elif q_part == 'b':
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

    parameters = {'max_depth':[3, 5, 8,14], 'min_samples_split': [4,5,6,9], 'min_samples_leaf': [4,5,6,10]}

    clf = tree.DecisionTreeClassifier()
    tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
    tree_clf = tree_clf.fit(arrX,arrY)
    tree_clf=tree_clf.best_estimator_
    out += str(tree_clf) + '\n'

    tree_clf = tree_clf.fit(arrX,arrY)

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=600)
    tree.plot_tree(tree_clf,feature_names = features, class_names = class_names, filled=True, rounded = True)
    fig.savefig('q1.1b.png')

    ypred = tree_clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = tree_clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = tree_clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    output_file = open(output_path + '/1_b.txt','w')
    output_file.write(out)

elif q_part == 'c':
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


    clf = tree.DecisionTreeClassifier(random_state=0)
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
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(arrX, arrY)
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

    train_scores = [clf.score(arrX, arrY) for clf in clfs]
    val_scores = [clf.score(val_arrX, val_arrY) for clf in clfs]
    test_scores = [clf.score(test_arrX, test_arrY) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training, validation and test sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, val_scores, marker="o", label="validation", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    fig.savefig('accuracy vs alpha.png')


    #best tree is one with highest validation accuracy 
    #ccp_alpha = 0.015

    clf = tree.DecisionTreeClassifier(ccp_alpha = 0.015)
    out += str(clf) + '\n'
    clf.fit(arrX,arrY)

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=600)
    tree.plot_tree(clf,feature_names = features, class_names = class_names, filled=True, rounded = True)
    fig.savefig('q1.1c.png')


    ypred = clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    output_file = open(output_path + '/1_c.txt','w')
    output_file.write(out)

elif q_part == 'd':
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

    parameters = {'max_features': [2,3,4], 'min_samples_split': [5,6,7], 'n_estimators': [90,100,120], 'oob_score': [True]}

    clf = ensemble.RandomForestClassifier()
    tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
    tree_clf = tree_clf.fit(arrX,arrY)
    tree_clf=tree_clf.best_estimator_
    out += str(tree_clf) + '\n'

    tree_clf.fit(arrX,arrY)

    ypred = tree_clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    out += "Out of bag accuracy : " + str(tree_clf.oob_score_) + '\n'
    val_ypred = tree_clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = tree_clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    output_file = open(output_path + '/1_d.txt','w')
    output_file.write(out)

elif q_part == 'e':
    train_data_o = pd.read_csv(train_path).to_numpy()
    val_data_o = pd.read_csv(val_path).to_numpy()
    test_data_o = pd.read_csv(test_path).to_numpy()


    train_data_o[train_data_o == '?'] = np.nan

    val_data_o[val_data_o == '?'] = np.nan

    test_data_o[test_data_o == '?'] = np.nan

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

    out += "imputer strategy is mean\n"
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(arrX,arrY)

    ypred = clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'


    tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
    tree_clf = tree_clf.fit(arrX,arrY)
    tree_clf=tree_clf.best_estimator_
    out += str(tree_clf) + '\n'

    tree_clf = tree_clf.fit(arrX,arrY)

    ypred = tree_clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = tree_clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = tree_clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    clf = tree.DecisionTreeClassifier(ccp_alpha = 0.015)
    out += str(clf) + '\n'
    clf.fit(arrX,arrY)

    ypred = clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    parameters = {'max_features': [2,3,4], 'min_samples_split': [5,6,7], 'n_estimators': [90,100,120]}

    clf = ensemble.RandomForestClassifier()
    tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
    tree_clf = tree_clf.fit(arrX,arrY)
    tree_clf=tree_clf.best_estimator_
    out += str(tree_clf) + '\n'

    tree_clf.fit(arrX,arrY)

    ypred = tree_clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = tree_clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = tree_clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

####################################################### MEDIAN###########################
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

    parameters = {'max_depth':[3,5], 'min_samples_split': [4,5,6], 'min_samples_leaf': [4,5,6]}


    out += "imputer strategy is median\n"
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(arrX,arrY)

    ypred = clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'


    tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
    tree_clf = tree_clf.fit(arrX,arrY)
    tree_clf=tree_clf.best_estimator_
    out += str(tree_clf) + '\n'

    tree_clf = tree_clf.fit(arrX,arrY)

    ypred = tree_clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = tree_clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = tree_clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    clf = tree.DecisionTreeClassifier(ccp_alpha = 0.015)
    out += str(clf) + '\n'
    clf.fit(arrX,arrY)

    ypred = clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    parameters = {'max_features': [2,3,4], 'min_samples_split': [5,6,7], 'n_estimators': [90,100,120]}

    clf = ensemble.RandomForestClassifier()
    tree_clf = GridSearchCV(estimator=clf, param_grid=parameters)
    tree_clf = tree_clf.fit(arrX,arrY)
    tree_clf=tree_clf.best_estimator_
    out += str(tree_clf) + '\n'

    tree_clf.fit(arrX,arrY)

    ypred = tree_clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = tree_clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = tree_clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    output_file = open(output_path + '/1_e.txt','w')
    output_file.write(out)

elif q_part == 'f':
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
    out += str(tree_clf)  + '\n'

    tree_clf = tree_clf.fit(arrX,arrY)

    ypred = tree_clf.predict(arrX)
    train_acc = np.sum(ypred == arrY)/len(arrY)
    out += "Training accuracy : " + str(train_acc) + '\n'

    val_ypred = tree_clf.predict(val_arrX)
    val_acc = np.sum(val_ypred == val_arrY)/len(val_arrY)
    out += "Validation accuracy : " + str(val_acc) + '\n'

    test_ypred = tree_clf.predict(test_arrX)
    test_acc = np.sum(test_ypred == test_arrY)/len(test_arrY)
    out += "Test accuracy : " + str(test_acc) + '\n'

    output_file = open(output_path + '/1_f.txt','w')
    output_file.write(out)

output_file.close()
