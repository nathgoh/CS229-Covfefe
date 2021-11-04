from time import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, train_test_split


def svm_baseline(C = [1, 10, 100, 1000], gamma = [0.0001, 0.001, 0.005], kernel = ['rbf'], split = 0.8):
    param_grid = {
        'C': C,
        'gamma': gamma,
        'kernel': kernel
    }

    df = pd.read_pickle('image_classified_df.pkl')
    df = df.astype(np.float32)
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1].to_numpy()

    print("Loaded RoCoLe Dataset. \n")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = split, random_state = 83, stratify = Y)
    
    # n_components = 800
    # print("Extracting top {} eigenleaves from {} coffee leaves".format(n_components, X_train.shape[0]))
    # t0 = time()
    # pca = PCA(n_components = n_components, svd_solver = 'randomized', whiten = True).fit(X_train)
    # X_train_pca = pca.transform(X_train)
    # X_test_pca = pca.transform(X_test)
    # print("Done in {}s \n".format(time() - t0))

    print("Training SVM baseline model....")
    t0 = time()
    svc = svm.SVC(probability = True)
    model = GridSearchCV(svc, param_grid, verbose = 2)

    model.fit(X_train, Y_train)

    print("Training completed in {}s \n".format(time() - t0))

    print("Predicting....")
    t0 = time()
    predictions = model.predict(X_test)

    print(Y_test)
    print(predictions)

    print("Prediction completed in {}s \n".format(time() - t0))
    print("Calculating metrics....")
    accuracy = metrics.accuracy_score(Y_test, predictions)
    # f1_score = metrics.f1_score(Y_test, predictions, average = 'weighted')
    # precision = metrics.precision_score(Y_test, predictions, average = 'weighted')
    # recall = metrics.recall_score(Y_test, predictions, average = 'weighted')

    print("Results from Grid Search: ")
    print("\n The best estimator across ALL searched params {}".format(model.best_estimator_))
    print("\n The best parameters across ALL searched params: {}".format(model.best_params_))
    print("\nAccuracy: {}".format(accuracy))

    print("\nClassification report: ")
    print(metrics.classification_report(Y_test, predictions, labels = [0, 1, 2, 3, 4, 5]))

    confusion = metrics.confusion_matrix(Y_test, predictions, labels = [0, 1, 2, 3, 4, 5])
    print("\n Confusion matrix:")
    print(confusion)
    cm = sns.heatmap(confusion, annot = True, fmt = 'd')
    plt.show()
    
