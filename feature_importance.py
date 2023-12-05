import random
import numpy as np
import pandas as pd
from utils.dataset import load_data
from utils.F1_score import F1_score
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE
import shap

np.random.seed(146)
random.seed(146)
X, y, X_test, y_test = load_data("data/train.csv")
feature_name = [key for key in X]

# Construct the train and val split
n_data = y.size
split = int(0.8 * n_data) # We use 0.8 of data to train.

X_train = X[:split]
y_train = y[:split]

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
n_train = y_train.size
n_val = y_val.size

# Linear Classifier
model = SGDClassifier(loss='log_loss', verbose=0)

# Decision Tree Classifier
# model = DecisionTreeClassifier(random_state=146)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = np.sum(y_test == y_pred) / len(y_test)
F1 = F1_score(y_pred, y_test)
print('Test:')
print('F1_score 0:', F1[0])
print('F1_score 1:', F1[1])
print('F1_Avg:', sum(F1)/2)
print('Accuracy:', acc)

# Permutation Importance
r = permutation_importance(model, X_val, y_val,
                            n_repeats=30,
                            random_state=146)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f"{feature_name[i] }",
              f"{r.importances_mean[i]:.3f}",
              f" +/- {r.importances_std[i]:.3f}")

# SHAP
X_train_summary = shap.kmeans(X_train, 10)
ex = shap.KernelExplainer(model.predict_proba, X_train_summary)

shap_values = ex.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

feature_enhance = np.mean(np.abs(shap_values[0]), axis=0)
np.save('linear.npy', feature_enhance)

# Feature Enhancement
feature_enhance = np.load('linear.npy')
for i, k in enumerate(X_train):
    X_train[k] *= (1+feature_enhance[i])
    X_test[k] *= (1+feature_enhance[i])
    X_val[k] *= (1+feature_enhance[i])

model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_val)

# Test
y_pred = model.predict(X_test)

acc = np.sum(y_test == y_pred) / len(y_test)
F1 = F1_score(y_pred, y_test)
print('Test:')
print('F1_score 0:', F1[0])
print('F1_score 1:', F1[1])
print('F1_Avg:', sum(F1)/2)
print('Accuracy:', acc)
