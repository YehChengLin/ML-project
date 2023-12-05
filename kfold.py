import random
import numpy as np
import pandas as pd
from utils.dataset import load_data
from utils.F1_score import F1_score
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

np.random.seed(146)
random.seed(146)
X, y, X_test, y_test = load_data("data/train.csv")
# Construct the train and val split
n_data = y.size

for k in [3,5,10]:
    F1_val = []
    F1_test = []
    y_pred = np.zeros((y_test.shape[0], 2))

    for fold in range(k):
        train_mask = np.ones(n_data, dtype=int)
        train_mask[int(fold * n_data/k):int((fold+1) * n_data/k)] = 0

        # split = int(0.8 * n_data) # We use 0.8 of data to train.

        X_train = X[train_mask==1]
        y_train = y[train_mask==1]
        # print('number of training data:', y_train.size)

        X_val = X[train_mask==0]
        y_val = y[train_mask==0]

        # Balance the positive sample
        n_1 = len(y_train[y_train==1])
        n_0 = len(y_train[y_train==0])
        # print('Positive count:', n_1, 'Negative count:', n_0)

        # SMOTE
        sampling_strategy = {1: max(n_0, n_1)}

        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=146)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        # print(len(y_train[y_train==1]))
        n_train = y_train.size
        n_val = y_val.size

        # Models
        
        # model = SGDClassifier(loss='log_loss', verbose=0)
        # model = KNeighborsClassifier(n_neighbors=11, weights='uniform', metric = 'minkowski', p=1)
        # model = KNeighborsClassifier(n_neighbors=11, weights='uniform', metric = 'minkowski', p=2)
        model = KNeighborsClassifier(n_neighbors=11, weights='uniform', metric = 'cosine')
        # model = DecisionTreeClassifier(random_state=146)
        # model = DecisionTreeClassifier(random_state=146, ccp_alpha=0.001)

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Ensemble the results for each fold
        y_pred += model.predict_proba(X_test)

    # Compute the final output
    y_pred = np.argmax(y_pred, axis=1)
    F1 = F1_score(y_pred, y_test)
    print(sum(F1)/2)

print(sum(F1_test)/k)
print(F1_val)
print(F1_test)
print(np.std(F1_val))
print(np.std(F1_test))