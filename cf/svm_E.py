import math
import random


class SVMClassifier:
    kernels = {'linear': lambda x, y, _: sum(xi * yi for xi, yi, in zip(x, y)),
               'polynomial': lambda x, y, d: (sum(xi * yi for xi, yi, in zip(x, y)) + 1) ** d,
               'gaussian': lambda x, y, sigma: math.exp(
                   -math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)) ** 2 / (2 * sigma ** 2)))
               }

    def __init__(self, C, kernel, kernel_param=None):
        self.kernel = self.kernels[kernel]
        self.kernel_param = kernel_param
        self.tolerance = 1e-3
        self.C = C

    def count_lambdas(self, K, y_train):
        lambdas = [0 for _ in range(len(y_train))]
        b = 0

        def get_prediction(i):
            return sum(yi * lambdas_i * k_i_j for yi, lambdas_i, k_i_j in zip(y_train, lambdas, K[i])) + b

        for _ in range(self.epochs_count):
            for i in range(len(y_train)):
                E_i = get_prediction(i) - y_train[i]
                if y_train[i] * E_i < -self.tolerance and lambdas[i] < self.C or y_train[i] * E_i > self.tolerance and \
                        lambdas[i] > 0:
                    j = random.choice([j for j in range(i)] + [j for j in range(i + 1, len(y_train))])
                    E_j = get_prediction(j) - y_train[j]

                    old_lambda_i = lambdas[i]
                    old_lambda_j = lambdas[j]

                    if y_train[i] != y_train[j]:
                        L = max(0.0, lambdas[j] - lambdas[i])
                        H = min(self.C, self.C + lambdas[j] - lambdas[i])
                    else:
                        L = max(0.0, lambdas[i] + lambdas[j] - self.C)
                        H = min(self.C, lambdas[i] + lambdas[j])

                    if abs(L - H) < 1e-15:
                        continue

                    eta = 2 * K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0:
                        continue

                    lambdas[j] -= y_train[j] * (E_i - E_j) / eta
                    lambdas[j] = min(max(lambdas[j], L), H)
                    if abs(lambdas[j] - old_lambda_j) < self.tolerance:
                        continue

                    lambdas[i] += y_train[i] * y_train[j] * (old_lambda_j - lambdas[j])

                    b1 = b - E_i - y_train[i] * (lambdas[i] - old_lambda_i) * K[i][i] - y_train[j] * (
                            lambdas[j] - old_lambda_j) * K[i][j]
                    b2 = b - E_j - y_train[i] * (lambdas[i] - old_lambda_i) * K[i][j] - y_train[j] * (
                            lambdas[j] - old_lambda_j) * K[j][j]
                    if 0 < lambdas[i] < self.C:
                        b = b1
                    elif 0 < lambdas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
        self.lambdas = lambdas
        self.b = b

    def fit(self, X_train, y_train, epochs_count=100):
        self.epochs_count = epochs_count
        self.X_train = X_train
        self.y_train = y_train
        K = [[self.kernel(x1, x2, self.kernel_param) for x2 in X_train] for x1 in X_train]
        self.count_lambdas(K, y_train)

    def get_params(self):
        return self.lambdas, self.b

    def predict(self, X_test):
        k = [self.kernel(X_test, xi, self.kernel_param) for xi in self.X_train]
        return math.copysign(1, sum(y_i * k_i * l_i for y_i, k_i, l_i in zip(self.y_train, k, self.lambdas)) + self.b)


if __name__ == "__main__":
    C = 1
    kernel = 'linear'
    n = int(input())
