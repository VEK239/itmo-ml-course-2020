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
    print(np.corrcoef(x, y)[0][1])
