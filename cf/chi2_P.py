from collections import defaultdict

if __name__ == "__main__":
    x_count, y_count = map(int, input().split())
    matrix = defaultdict(int)
    x_sum_count = [0 for _ in range(x_count)]
    y_sum_count = [0 for _ in range(y_count)]
    n = int(input())
    for i in range(n):
        x, y = map(int, input().split())
        matrix[(x - 1, y - 1)] += 1
        x_sum_count[x - 1] += 1
        y_sum_count[y - 1] += 1
    chi2 = n
    for (i, j) in matrix.keys():
        expected = y_sum_count[j] * x_sum_count[i] / n
        chi2 += (expected - matrix[(i, j)]) ** 2 / expected - expected
    print(chi2)
