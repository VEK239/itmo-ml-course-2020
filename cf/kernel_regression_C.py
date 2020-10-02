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


if __name__ == "__main__":
    n, m = map(int, input().split())
    data = [list(map(int, input().split())) for _ in range(n)]
    X_train = [line[:-1] for line in data]
    y_train = [[line[-1]] for line in data]
    X_test = list(map(int, input().split()))

    classifier = KNNRegression(input(), input(), input(), int(input()))
    classifier.fit(X_train, y_train)
    print(classifier.predict(X_test)[0])
