def compute_positive(conf_matrix):
    return [sum(line) for line in conf_matrix]


def compute_true_positive(conf_matrix):
    return [line[i] for i, line in enumerate(conf_matrix)]


def compute_false_positive(conf_matrix):
    return [sum(line_1[i] for line_1 in conf_matrix) - line[i] for i, line in enumerate(conf_matrix)]


def compute_f1_score(conf_matrix, f1_type="default", prec=None, rec=None):
    true_positive = compute_true_positive(conf_matrix)
    false_positive = compute_false_positive(conf_matrix)
    positive = compute_positive(conf_matrix)
    precision = compute_precision(true_positive, false_positive)
    recall = compute_recall(true_positive, positive)
    all = sum(positive)
    if f1_type == 'default':
        return [0 if p + r == 0 else 2 * p * r / (p + r) for p, r in zip(prec, rec)]
    elif f1_type == 'macro':
        class_probabilities = [p / all for p in positive]
        weighted_precision = sum([cp * p for cp, p in zip(class_probabilities, precision)])
        weighted_recall = sum([cp * r for cp, r in zip(class_probabilities, recall)])
        return compute_f1_score(conf_matrix, "default", [weighted_precision], [weighted_recall])[0]
    elif f1_type == 'micro':
        return sum(
            f1 * p for f1, p in zip(compute_f1_score(conf_matrix, "default", precision, recall), positive)) / all


def compute_precision(true_positive, false_positive):
    return [0 if tp + fp == 0 else tp / (tp + fp) for tp, fp in zip(true_positive, false_positive)]


def compute_recall(true_positive, positive):
    return [0 if p == 0 else tp / p for tp, p in zip(true_positive, positive)]

from math import *
from statistics import mean

distance_names = ["manhattan", "euclidean", "chebyshev"]
kernel_names = ["uniform", "triangular", "epanechnikov", "quartic", "triweight", "tricube", "gaussian", "cosine",
                "logistic",
                "sigmoid"]
window_names = ["fixed", "variable"]

check_absolute_1 = lambda x, val: 0 if abs(x) >= 1 else val
dist_function = {"manhattan": lambda x, y: sum(abs(x_i - y_i) for x_i, y_i in zip(x, y)),
                 "euclidean": lambda x, y: sqrt(sum((x_i - y_i) ** 2 for x_i, y_i in zip(x, y))),
                 "chebyshev": lambda x, y: max(abs(x_i - y_i) for x_i, y_i in zip(x, y))}
kernel_function = {"uniform": lambda x: check_absolute_1(x, 1 / 2),
                   "triangular": lambda x: check_absolute_1(x, 1 - abs(x)),
                   "epanechnikov": lambda x: check_absolute_1(x, 3 / 4 * (1 - x ** 2)),
                   "quartic": lambda x: check_absolute_1(x, 15 / 16 * (1 - x ** 2) ** 2),
                   "triweight": lambda x: check_absolute_1(x, 35 / 32 * (1 - x ** 2) ** 3),
                   "tricube": lambda x: check_absolute_1(x, 70 / 81 * (1 - abs(x) ** 3) ** 3),
                   "gaussian": lambda x: 1 / (sqrt(2 * pi)) * e ** (-1 / 2 * x ** 2),
                   "cosine": lambda x: check_absolute_1(x, pi / 4 * cos(pi / 2 * x)),
                   "logistic": lambda x: 1 / (e ** x + 2 + e ** (-x)),
                   "sigmoid": lambda x: 2 / pi / (e ** x + e ** (-x))}
window_function = {"fixed": lambda d, r: r,
                   "variable": lambda d, kn: sorted(d)[kn]}


class KNNRegression:
    def __init__(self, dist_fun, kernel_fun, window_fun, h):
        self.dist_f = dist_function[dist_fun]
        self.kernel_f = kernel_function[kernel_fun]
        self.window_f = window_function[window_fun]
        self.h = h

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, x, target_count=1):
        distances = [self.dist_f(line, x) for line in self.X_train]
        win_result = self.window_f(distances, self.h)
        weighted_y = [0 for _ in range(target_count)]
        weights_sum = 0
        for i, distance in enumerate(distances):
            weight = self.kernel_f(distance / win_result) if win_result != 0 else 0
            for j in range(target_count):
                weighted_y[j] += weight * self.y_train[i][j]
            weights_sum += weight
        predicted = []
        for j in range(target_count):
            predicted.append(
                weighted_y[j] / weights_sum if weights_sum != 0 else mean(line[j] for line in self.y_train))
        return predicted


#%% md

# ITMO ML Lab1. Kernel regression.

#%%

import pandas as pd

#%% md

## Data preprocessing

#%%

penguins = pd.read_csv("../data/penguins.csv")

#%%

penguins

#%%

penguins = penguins.dropna()

#%%

penguins

#%%

penguins = penguins.reset_index().drop(columns=['index'])
penguins

#%%

y = penguins.species

#%%

penguins['fromTorgersen'] = pd.Series(penguins.island == 'Torgersen')
penguins['fromBiscoe'] = pd.Series(penguins.island == 'Biscoe')
penguins['fromDream'] = pd.Series(penguins.island == 'Dream')
penguins['sex'] = pd.Series(penguins.sex == 'FEMALE')
X = penguins.drop(columns=['species', 'island'])

#%%

y

#%%

y = y.replace(to_replace=['Adelie', 'Chinstrap', 'Gentoo'], value=[1, 2, 3])

#%%

y

#%%

X

#%%

X.dtypes

#%%

X = X.astype({'sex': 'int32', 'fromTorgersen': 'int32', 'fromBiscoe': 'int32','fromDream': 'int32'})

#%% md

## Normalizing

#%%

def minmax(data):
    value_min = min(data)
    value_max = max(data)
    return value_min, value_max


def normalize(series):
    data = list(series)
    v_min, v_max = minmax(data)
    delta = v_max - v_min
    for i in range(len(data)):
        data[i] = (data[i] - v_min) / delta
    return pd.Series(data)

#%%

X.culmen_length_mm = normalize(penguins.culmen_length_mm)
X.culmen_depth_mm = normalize(penguins.culmen_depth_mm)
X.flipper_length_mm = normalize(penguins.flipper_length_mm)
X.body_mass_g = normalize(penguins.body_mass_g)

#%%

X.head()

#%% md

## Hyperparameter optimization

#%%

import sys
sys.path.append('../../../cf')
from uniform_split_A import uniform_split
from kernel_regression_C import KNNRegression, distance_names, kernel_names, window_names
from f1_score_B import compute_f1_score

#%%

def one_out_cv(X, y, class_recognizer, classes_count, params, target_count=1):
    y_test = []
    y_predicted = []
    for i in range(len(X)):
        X_copy = X.copy()
        y_copy = y.copy()
        x_test = X_copy.pop(i)
        y_test.append(class_recognizer(y_copy.pop(i)))
        classifier = KNNRegression(*params)
        classifier.fit(X_copy, y_copy)
        y_predicted.append(class_recognizer(classifier.predict(x_test, target_count)))
    return get_confusion_matrix(y_test, y_predicted, classes_count)

def get_confusion_matrix(y_test, y_predicted, classes_count):
    confusion_matrix = [[0 for _ in range(classes_count)] for _ in range(classes_count)]
    for test, pred in zip(y_test, y_predicted):
        confusion_matrix[test - 1][pred - 1] += 1
    return confusion_matrix

def hyperparameter_optimization(X, y, classes_count, class_recognizer, target_count=1, curr_window_names=window_names):
    best_f1_score = 0
    for distance_name in distance_names:
        for kernel_name in kernel_names:
            for window_name in curr_window_names:
                for h in range(1, 40, 2):
                    params = [distance_name, kernel_name, window_name, h]
                    confusion_matrix = one_out_cv(X, y, class_recognizer, classes_count, params, target_count)
                    f1_score = compute_f1_score(confusion_matrix, 'macro')
                    if best_f1_score < f1_score:
                        best_f1_score = f1_score
                        best_params = params
                        print("Finished with best params", best_params, f1_score)
    print('Overall best params:', best_params)
    print('Overall best macro f1-score:', best_f1_score)
    return best_params, best_f1_score

#%% md

## Naive representation of target feature

#%%

y

#%%

X

#%%

X = X.values.tolist()
y = list([i] for i in y)

#%%

import math

#%%

hyperparameter_optimization(X, y, 3, lambda x: round(x[0]))

#%% md

## One-hot representation

#%%

import numpy as np

#%%

y = [i[0] for i in y]
y = pd.DataFrame({'target': y})
y

#%%

y['target0'] = pd.Series(y.target == 1)
y['target1'] = pd.Series(y.target == 2)
y['target2'] = pd.Series(y.target == 3)
y = y.drop(columns = ['target'])
y = y.astype({'target0': 'int32', 'target1': 'int32', 'target2': 'int32'})

#%%

y = y.values.tolist()

#%%

hyperparameter_optimization(X, y, 3, lambda x: np.asarray(x).argmax() + 1, 3)

#%% md

## Analyzing best model

#%%

import seaborn as sns

#%%

def plot(neighbours_count, f1_score, metrics_name):
    neighbours_count = pd.Series(neighbours_count)
    f1_score = pd.Series(f1_score)
    plot_data = pd.DataFrame({"neighbours_count": neighbours_count, metrics_name: f1_score})
    sns.lineplot(data=plot_data, x="neighbours_count", y=metrics_name)

#%% md

### Variable window size

#%%

macro_f1_scores = []
micro_f1_scores = []
matrices = []
class_recognizer = lambda x: np.asarray(x).argmax() + 1
for h in range(1, 50):
    params = ['euclidean', 'uniform', 'variable', h]
    confusion_matrix = one_out_cv(X, y, class_recognizer, 3, params, 3)
    matrices.append(confusion_matrix)
    macro_f1_score = compute_f1_score(confusion_matrix, 'macro')
    macro_f1_scores.append(macro_f1_score)

#%%

plot([h for h in range(1, 50)], macro_f1_scores, 'macro_f1_score')

#%% md

### Fixed window size

#%%

hyperparameter_optimization(X, y, 3, lambda x: np.asarray(x).argmax() + 1, 3, curr_window_names=['fixed'])

#%%

macro_f1_scores = []
matrices = []
class_recognizer = lambda x: np.asarray(x).argmax() + 1
for h in range(1, 50):
    params = ['manhattan', 'triweight', 'fixed', h]
    confusion_matrix = one_out_cv(X, y, class_recognizer, 3, params, 3)
    matrices.append(confusion_matrix)
    macro_f1_score = compute_f1_score(confusion_matrix, 'macro')
    macro_f1_scores.append(macro_f1_score)

#%%

plot([h for h in range(1, 50)], macro_f1_scores, 'macro_f1_score')

#%%
