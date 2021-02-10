if __name__ == "__main__":
    k = int(input())
    n = int(input())
    xs = []
    ys = []
    for i in range(n):
        x, y = map(int, input().split())
        xs.append(x)
        ys.append(y)
    inner = 0
    outer = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ys[i] == ys[j]:
                inner += abs(xs[i] - xs[j])
            else:
                outer += abs(xs[i] - xs[j])
    print(inner * 2)
    print(outer * 2)
