import math
import operator
from collections import defaultdict
import numpy as np


class BayesClassifier:
    def __init__(self, alpha, gram_len, lambdas):
        self.classes = defaultdict(int)
        self.frequencies = defaultdict(int)
        self.probabilities = defaultdict(int)
        self.grams = set()
        self.alpha = alpha
        self.gram_len = gram_len
        self.lambdas = lambdas
        self.log_probs = defaultdict(float)
        self.back_log_probs = defaultdict(float)

    def create_n_grams(self, labels, lines):
        for label, line in zip(labels, lines):
            line_grams = set()
            self.classes[label] += 1
            for i in range(len(line) - self.gram_len + 1):
                gram = ' '.join(line[i: i + self.gram_len])
                self.grams.add(gram)
                line_grams.add(gram)
            for gram in line_grams:
                self.frequencies[label, gram] += 1

    def create_conditional_probs(self):
        # p(w_i|c_j)
        for label in self.classes.keys():
            for gram in self.grams:
                self.probabilities[gram, label] = (self.frequencies[label, gram] + self.alpha) / (
                        self.classes[label] + self.alpha * 2)
                self.log_probs[gram, label] = math.log(self.probabilities[gram, label])
                self.back_log_probs[gram, label] = math.log(1 - self.probabilities[gram, label])

    def fit(self, X, y):
        self.create_n_grams(y, X)
        self.create_conditional_probs()

    def predict(self, X, return_all=False):
        resulting_probs = {}
        grams = [' '.join(X[i: i + self.gram_len]) for i in range(len(X) - self.gram_len + 1)]
        for label in self.classes.keys():
            resulting_probs[label] = math.log(self.classes[label] * self.lambdas[label - 1]) + \
                                     self.get_back_conditional_probs(label, grams)

        if return_all:
            for label, prob in resulting_probs.items():
                resulting_probs[label] = pow(math.e, prob)
            sum_prob = sum(resulting_probs.values())
            for label, prob in resulting_probs.items():
                resulting_probs[label] = prob / sum_prob
            return resulting_probs
        else:
            lp1 = resulting_probs[1]
            lp2 = resulting_probs[2]
            p2 = 1 / (1 + np.exp(lp1 - lp2))
            p1 = 1 - p2
            return [max(resulting_probs.items(), key=operator.itemgetter(1))[0], p1, p2]

    def get_back_conditional_probs(self, label, grams):
        prob = 0
        for w in self.grams:
            if w in grams:
                prob += self.log_probs[w, label]
            else:
                prob += self.back_log_probs[w, label]
        return prob


if __name__ == "__main__":
    n = int(input())
    lambdas = list(map(int, input().split()))
    alpha = int(input())
    train_count = int(input())
    X, y = [], []
    for i in range(train_count):
        line = input().split()
        X.append(line[2:])
        print(X)
        y.append(int(line[0]))
    classifier = BayesClassifier(alpha, 1, lambdas)
    classifier.fit(X, y)
    test_count = int(input())
    for i in range(test_count):
        print(' '.join(map(str, classifier.predict(input().split()[1:], return_all=True).values())))
