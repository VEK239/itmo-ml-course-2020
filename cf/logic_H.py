if __name__ == "__main__":
    args_count = int(input())
    f = []
    for _ in range(2 ** args_count):
        f += [int(input())]
    ones_count = sum(f)
    if ones_count > 0:
        print(2)
        print(ones_count, 1)
        for i in range(2 ** args_count):
            if f[i] != 0:
                b = 0.5
                k = i
                for j in range(args_count):
                    print(1 if k % 2 else -1, end=' ')
                    b -= k % 2
                    k //= 2
                print(b)
        print(*(1 for _ in range(ones_count)), -0.5)
    else:
        print(1, args_count + 1, ' '.join('-1' for _ in range(args_count + 1)))
