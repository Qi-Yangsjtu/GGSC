import numpy as np

def get_median_idx(X, idxs, feature):
    n = len(idxs)
    k = n // 2
    col = map(lambda i: (i, X[i][feature]), idxs)
    sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x: x[1]))
    media_idx = list(sorted_idxs)[k]
    return media_idx


def get_variance(X, idxs, feature):
    n = len(idxs)
    col_sum = col_sum_sqr = 0
    for idx in idxs:
        xi = X[idx][feature]
        col_sum += xi
        col_sum_sqr += xi ** 2
    return col_sum_sqr / n - (col_sum / n) ** 2


def choose_feature(X, idxs):
    m = len(X[0])
    variances = map(lambda j: (j, get_variance(X, idxs, j)), range(m))
    a = max(variances, key=lambda x: x[1])[0]
    return a

def split_feature(X, idxs, feature, median_idx):
    idxs_split = [[], []]
    split_val = X[median_idx][feature]
    for idx in idxs:
        if idx == median_idx:
            idxs_split[0].append(idx)
            continue
        xi = X[idx][feature]
        if xi < split_val:
            idxs_split[0].append(idx)
        else:
            idxs_split[1].append(idx)
    return idxs_split

def split_by_kdtree(points, threshold):
    candidate_list = []
    cur_list=[]
    point = points
    num = point.shape[0]
    num_this_group = point.shape[0]
    idxs = range(num_this_group)
    if num>threshold:
        cur_list.append(idxs)
    else:
        candidate_list.append(idxs)
    while len(cur_list):
        this_idxs = cur_list.pop(0)
        dim = choose_feature(point, this_idxs)
        median_id = get_median_idx(point, this_idxs, dim)
        idxs_left, idxs_right = split_feature(point, this_idxs, dim, median_id)
        if len(idxs_left)>threshold:
            cur_list.append(idxs_left)
        else:
            candidate_list.append(idxs_left)
        if len(idxs_right)>threshold:
            cur_list.append(idxs_right)
        else:
            candidate_list.append(idxs_right)
    return candidate_list