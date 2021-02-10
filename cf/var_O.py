if __name__ == "__main__":
    x_count = int(input())
    n = int(input())
    ey2 = [0 for i in range(x_count)]
    ey = [0 for i in range(x_count)]
    prob = [0 for i in range(x_count)]
    for i in range(n):
        x, y = map(int, input().split())
        ey2[x - 1] += y * y / n
        ey[x - 1] += y / n
        prob[x - 1] += 1 / n
    var_y2 = sum(ey2)
    var2_y = sum(ey[i] * ey[i] / prob[i] if prob[i] != 0 else 0 for i in range(x_count))
    print(var_y2 - var2_y)