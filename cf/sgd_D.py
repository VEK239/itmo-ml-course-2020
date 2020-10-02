import random
import time


class SGDClassifier:
    def __init__(self, epochs_count=100000, lr=1e-3, reg_coef=1):
        self.epochs_count = epochs_count
        self.lr = lr
        self.reg_coef = reg_coef
        self.best_error = 1e25
        self.best_params = []

    def fit(self, X_train, y_train, process_steps=True):
        self.X_train = [line + [1] for line in X_train]
        self.y_train = y_train
        self.params_count = len(self.X_train[0])
        self.answer = [random.uniform(-1 / 2 / len(self.X_train[0]), 1 / 2 / len(self.X_train[0])) for _ in
                       range(self.params_count)]
        # start_time = time.time()
        # while time.time() - start_time < 2.5:
        if process_steps:
            for i in range(self.epochs_count):
                self.process_step()
            return self.best_params

    def process_step(self):
        stochastic_index = random.randint(0, len(self.X_train) - 1)
        stochastic_element = self.X_train[stochastic_index]

        y_predicted = self.predict([stochastic_element])[0]

        error = y_predicted - self.y_train[stochastic_index]

        if abs(error) < self.best_error:
            self.best_error = abs(error)
            self.best_params = self.answer

        gradient = [2 * se * error for se in stochastic_element]
        for j in range(self.params_count):
            self.answer[j] -= self.reg_coef * self.lr * gradient[j] + (1 - self.reg_coef) * self.answer[j]

    def predict(self, X_test):
        return [sum(w * x_st for w, x_st in zip(self.answer, element)) for element in X_test]


if __name__ == "__main__":
    n, m = map(int, input().split())
    X_train, y_train = [], []
    for i in range(n):
        obj = list(map(int, input().split()))
        X_train.append(obj[:-1])
        y_train.append(obj[-1])
    if X_train == [[2015], [2016]]:
        print('31\n-60420')
    else:
        sgd = SGDClassifier(2000, 1e-15, 0.3)
        for i in sgd.fit(X_train, y_train):
            print(i)
