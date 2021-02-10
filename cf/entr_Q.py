from collections import defaultdict
from math import log

if __name__ == "__main__":
    x_count, y_count = map(int, input().split())
    p_x_y = defaultdict(int)
    p_x = [0 for _ in range(x_count)]
    n = int(input())
    for i in range(n):
        x, y = map(int, input().split())
        p_x[x - 1] += 1 / n
        p_x_y[(x - 1, y - 1)] += 1 / n
    entropy = sum(prob * (log(p_x[x]) - log(prob)) for (x, y), prob in p_x_y.items())
    print(entropy)
