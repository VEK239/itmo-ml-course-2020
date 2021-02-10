import numpy as np

if __name__ == "__main__":
    n = int(input())
    xs = []
    ys = []
    for i in range(n):
        x, y = map(int, input().split())
        xs.append(x)
        ys.append(y)
    x = np.asarray(xs)
    y = np.asarray(ys)
    order = x.argsort()
    x_ranks = order.argsort()
    order = y.argsort()
    y_ranks = order.argsort()
    print(np.corrcoef(x_ranks, y_ranks)[0][1])
