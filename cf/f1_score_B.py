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


if __name__ == "__main__":
    classes_count = int(input())
    confusion_matrix = [list(map(int, input().split())) for _ in range(classes_count)]
    macro_f1_score = compute_f1_score(confusion_matrix, "macro")
    micro_f1_score = compute_f1_score(confusion_matrix, "micro")
    print(macro_f1_score)
    print(micro_f1_score)
