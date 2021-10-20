import pandas as pd
import numpy as np
import cudf
from cuml import svm, metrics
from sklearn.model_selection import GridSearchCV, train_test_split


def svm_baseline(C = [0.1], gamma = [0.0001], kernel = ['rbf'], split = 0.8):
    param_grid = {
        'C': C,
        'gamma': gamma,
        'kernel': kernel
    }

    df = pd.read_pickle('image_classified_df.pkl')
    df = df.astype(np.float32)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    print("Loaded RoCoLe Dataset.")

    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = split, random_state = 83, stratify = Y)
    print("Training SVM baseline model....")
    svc = svm.SVC(probability = True)
    model = GridSearchCV(svc, param_grid, verbose = 2)

    model.fit(X_train, Y_train)

    print("Predicting....")
    predictions = model.predict(X_test)

    print("Calculating metrics....")
    accuracy = metrics.accuracy.accuracy_score(predictions, Y_test)

    print(accuracy)
    # return accuracy


