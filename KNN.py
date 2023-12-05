import random
import numpy as np
import pandas as pd
from utils.dataset import load_data
from utils.F1_score import F1_score
import sklearn 
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

np.random.seed(146)
random.seed(146)
X, y, X_test, y_test = load_data("data/train.csv")
# Construct the train and val split
n_data = y.size
split = int(0.8 * n_data) # We use 0.8 of data to train.

X_train = X[:split]
y_train = y[:split]
# print('number of training data:', y_train.size)

X_val = X[split:]
y_val = y[split:]

# Balance the positive sample
n_1 = len(y_train[y_train==1])
n_0 = len(y_train[y_train==0])
print('Positive count:', n_1, 'Negative count:', n_0)

# SMOTE
sampling_strategy = {1: max(n_0, n_1)}
sm = SMOTE(sampling_strategy=sampling_strategy, random_state=146)
X_train, y_train = sm.fit_resample(X_train, y_train)

print(len(y_train[y_train==1]))
n_train = y_train.size
n_val = y_val.size

# KNN Classifier
model = KNeighborsClassifier(n_neighbors=11, weights='uniform', metric = 'minkowski', p=1)
# model = KNeighborsClassifier(n_neighbors=11, weights='uniform', metric = 'minkowski', p=2)
# model = KNeighborsClassifier(n_neighbors=11, weights='uniform', metric = 'cosine')

# Fit the KNN model on the training data
model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_val)

acc = np.sum(y_val == y_pred) / n_val
F1 = F1_score(y_pred, y_val)
print('F1_score 0:', F1[0])
print('F1_score 1:', F1[1])
print('F1_Avg:', sum(F1)/2)
print('F1_WAvg:', (F1[0]*len(y_val[y_val==0])+F1[1]*len(y_val[y_val==1]))/len(y_val))
print('Accuracy:', acc)
print()

# Test
y_pred = model.predict(X_test)
df_output = pd.DataFrame()

acc = np.sum(y_test == y_pred) / len(y_test)
F1 = F1_score(y_pred, y_test)
print('Test:')
print('F1_score 0:', F1[0])
print('F1_score 1:', F1[1])
print('F1_Avg:', sum(F1)/2)
print('F1_WAvg:', (F1[0]*len(y_test[y_test==0])+F1[1]*len(y_test[y_test==1]))/len(y_test))
print('Accuracy:', acc)
