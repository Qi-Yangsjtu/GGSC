import numpy as np
from kdtree_util import split_by_kdtree
import matplotlib.pyplot as plt
import os

def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def split_point_cloud(gs, min_samples):
    point = gs.center
    splited_results = split_by_kdtree(point, min_samples)
    print("Num of sub GS: ", len(splited_results))
    labels = np.zeros(point.shape[0])
    id = 1
    points = 0
    while len(splited_results):
        this_sub = splited_results.pop(0)
        labels[this_sub] = id
        points+=len(this_sub)
        id+=1
    return labels

def plot_point_seg(gs, labels):
    unique_label = np.unique(labels)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in unique_label:
        if label == -1:
            color = 'black'
        else:
            color = plt.cm.jet(label/np.max(labels+1))
        cluster_points = gs.center[labels == label]
        ax.scatter(cluster_points[:,0],cluster_points[:,1],cluster_points[:,2],color = color)
    ax.set_title('Point Segmentation')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def get_decimal(number):
    decimal_part = str(number).split('.')[1]
    return len(decimal_part)

def GS_sort(center, normal, baseColor, R_SH, G_SH, B_SH, opacity, scale, rotate):
    sorted_index = np.lexsort((center[:,0], center[:,1], center[:,2]))
    sorted_center = center[sorted_index]

    sorted_normal = normal[sorted_index]
    sorted_baseColor = baseColor[sorted_index]
    sorted_R_SH = R_SH[sorted_index]
    sorted_G_SH = G_SH[sorted_index]
    sorted_B_SH = B_SH[sorted_index]
    sorted_opacity = opacity[sorted_index]
    sorted_scale = scale[sorted_index]
    sorted_rotate = rotate[sorted_index]
    return sorted_center, sorted_normal, sorted_baseColor, sorted_R_SH, sorted_G_SH, sorted_B_SH, sorted_opacity, sorted_scale, sorted_rotate

def get_DataRange(data):
    first = data[0]
    print(first.shape)
    max_value = data[0][0]
    min_value = data[0][0]
    print("max ini: ", max_value, min_value)

    for sub_list in data:
        sub_min_value = np.min(sub_list)
        print("sub_min", sub_min_value)
        sub_max_value = np.max(sub_list)
        print(sub_max_value)
        if sub_min_value < min_value:
            min_value = sub_min_value
        if sub_max_value > max_value:
            max_value = sub_max_value
    print("MAX and MIN: ", max_value, min_value)
    return max_value, min_value