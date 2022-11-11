import torch


def euclidean_distance(a, b):
    return torch.dist(a, b)


def KNN_clustering(k: int, flags: list, flags_class: list, test_cases: list, num_class=2):
    assert k <= len(flags), "Too many K, K must be less than the number of flags"

    result_list = [-1] * len(test_cases)

    for idx_case, data in enumerate(test_cases):
        distances = [euclidean_distance(data, flag) for flag in flags]

        result = []
        for idx, dist in enumerate(distances):
            result.append((dist, idx))
            if len(result) > k:
                result.sort()
                result.pop()

        class_list = [0] * num_class  # in a binary case -> [0, 0]
        for _, idx in result:
            class_list[flags_class[idx]] += 1

        result_list[idx_case] = class_list
        # result_list[idx_case] = class_list.index(max(class_list))

    return result_list
