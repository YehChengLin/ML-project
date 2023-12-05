import random
import numpy as np
import pandas as pd
from utils.dataset import load_data
from utils.F1_score import F1_score
import matplotlib.pyplot as plt
import sklearn 
from sklearn.tree import DecisionTreeClassifier
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
n_train = y_train.size
n_val = y_val.size


# Cost Complexity Pruning
clf = DecisionTreeClassifier(random_state=146)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print(len(ccp_alphas))

# For each alpha we will append our model to a list
clfs = []
alphas = ccp_alphas[::50].tolist() + ccp_alphas[-20:].tolist()
alphas.append(ccp_alphas[-1])
for ccp_alpha in alphas:
    clf = DecisionTreeClassifier(random_state=146, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Compute average F1 score vs alpha
val_F1avg = []
for c in clfs:
    y_pred = c.predict(X_val)
    F1 = F1_score(y_pred, y_val)
    val_F1avg.append(sum(F1)/2)
print(val_F1avg)

plt.scatter(alphas,val_F1avg)
plt.plot(alphas,val_F1avg,label='F1_avg',drawstyle="steps-post")
plt.legend()
plt.title('F1 vs alpha')
plt.savefig('FvA.png')

# Decision Tree Classifier
model = DecisionTreeClassifier(random_state=146, ccp_alpha=0.001)

# Fit the Decision Tree model on the training data
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
