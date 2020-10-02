def uniform_split(data, parts_count, classes_count):
    data = list(data)  # pandas case
    parts = [[] for _ in range(parts_count)]
    classes = [[] for _ in range(classes_count)]
    cur_part = 0
    for el_index, el_class in enumerate(data):
        classes[el_class - 1].append(el_index)
    for class_index, class_elements in enumerate(classes):
        for i, element in enumerate(class_elements):
            parts[cur_part].append(i)
            cur_part = (cur_part + 1) % parts_count
    return parts


if __name__ == "__main__":
    n, classes_count, parts_count = map(int, input().split())
    data = list(map(int, input().split()))
    parts = uniform_split(data, parts_count, classes_count)
    for part in parts:
        print(len(part), *(el for el in sorted([data[i] for i in part])))