import math
import os
import statistics

from sgd_D import SGDClassifier
from tqdm import tqdm


def read_dataset(index):
    with open('problem_D_tests/' + index, 'r') as inf:
        m = int(inf.readline())
        n = int(inf.readline())
        X_train, y_train = [], []
        for i in range(n):
            obj = list(map(int, inf.readline().split()))
            X_train.append(obj[:-1])
            y_train.append(obj[-1])
        n = int(inf.readline())
        X_test, y_test = [], []
        for i in range(n):
            obj = list(map(int, inf.readline().split()))
            X_test.append(obj[:-1])
            y_test.append(obj[-1])

    return X_train, y_train, X_test, y_test


def count_nrmse(y_predicted, y_test):
    try:
        return math.sqrt(sum((y_pred - y_t) ** 2 for y_pred, y_t in zip(y_predicted, y_test))) / len(y_test) / (
                max(y_test) - min(y_test))
    except:
        return 1


def count_smape(y_predicted, y_test):
    try:
        return sum(2 * abs(y_pred - y_t) / (abs(y_pred) + abs(y_t)) for y_pred, y_t in zip(y_predicted, y_test)) / len(
            y_test)
    except:
        return 2


def hyperparameter_optimization(datasets, Model):
    best_loss = 2
    for lr in [0.05, 0.1, 0.2, 0.3]:
        for reg_coef in [0, 0.2, 0.4]:
            for i in tqdm(range(10000, 100000, 10000)):
                params = [i, lr, reg_coef]
                loss = []
                for dataset in datasets:
                    model = Model(i, lr, reg_coef)
                    X_train, y_train, X_test, y_test = dataset[0], dataset[1], dataset[2], dataset[3]
                    model.fit(X_train, y_train)
                    y_predicted = model.predict(X_test)
                    nrmse = count_smape(y_predicted, y_test)
                    loss.append(nrmse)
                if statistics.mean(loss) < best_loss:
                    best_loss = statistics.mean(loss)
                    best_params = params
                    print("Finished with best params", best_params, best_loss)
        print('Finished lr', lr)
    print('Overall best params:', best_params)
    print('Overall best loss:', best_loss)
    return best_params, best_loss


if __name__ == "__main__":
    datasets = [read_dataset(dataset) for dataset in os.listdir('problem_D_tests')]
    hyperparameter_optimization(datasets, SGDClassifier)
