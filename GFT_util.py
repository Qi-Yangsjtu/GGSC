import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm
import concurrent.futures

def generate_graph_from_point(point, radius):
    vertices = point
    sigma = radius
    dis_matrix = distance_matrix(vertices, vertices)
    adj_matrix = np.exp(- dis_matrix / sigma)
    np.fill_diagonal(adj_matrix,0 )
    degree_matrix = np.diag(np.sum(adj_matrix,axis=1))
    L_matrix = degree_matrix - adj_matrix
    eigenvalues, eigenvectors = np.linalg.eig(L_matrix)
    indices = np.argsort(eigenvalues)
    # sorted_eigenvalues = eigenvalues[indices]
    sorted_eigenvectors = eigenvectors[indices]
    return sorted_eigenvectors

def color_GFT(gs, labels, label, eigenValueMatrix):
    Y_SH, U_SH, V_SH = gs.SH_RGB_To_SH_YUV()
    baseY, baseU, baseV = gs.baseRGB_To_baseYUV()

    sub_base_Y = baseY[labels == label]
    sub_base_U = baseU[labels == label]
    sub_base_V = baseV[labels == label]

    sub_SH_Y = Y_SH[labels == label, :]
    sub_SH_U = U_SH[labels == label, :]
    sub_SH_V = V_SH[labels == label, :]

    res_baseY = np.dot(np.transpose(sub_base_Y), eigenValueMatrix)
    res_baseU = np.dot(np.transpose(sub_base_U), eigenValueMatrix)
    res_baseV = np.dot(np.transpose(sub_base_V), eigenValueMatrix)
    res_SH_Y = np.dot(np.transpose(sub_SH_Y), eigenValueMatrix)
    res_SH_U = np.dot(np.transpose(sub_SH_U), eigenValueMatrix)
    res_SH_V = np.dot(np.transpose(sub_SH_V), eigenValueMatrix)

    res_combine_Y = np.vstack((res_baseY, res_SH_Y))
    res_combine_U = np.vstack((res_baseU, res_SH_U))
    res_combine_V = np.vstack((res_baseV, res_SH_V))

    return res_combine_Y, res_combine_U, res_combine_V

def opacity_GFT(gs, labels, label, eigenValueMatrix):
    opacity = gs.opacity
    sub_opacity = opacity[labels == label]
    res_opacity = np.dot(sub_opacity, eigenValueMatrix)
    return res_opacity

def scale_GFT(gs, labels, label, eigenValueMatrix):
    scale = gs.scale
    sub_scale = scale[labels == label]
    res_scale = np.dot(np.transpose(sub_scale), eigenValueMatrix)
    return res_scale

def rotate_GFT(gs, labels, label, eigenValueMatrix):
    rotate = gs.rotate
    sub_rotate = rotate[labels == label]
    res_rotate = np.dot(np.transpose(sub_rotate), eigenValueMatrix)
    return res_rotate

def graph_trans(gs, labels,radius):
    num_classes, counts = np.unique(labels, return_counts=True)
    res_Y = []
    res_U = []
    res_V = []
    res_opacity = []
    res_scale = []
    res_rotation = []
    for label, count in tqdm(zip(num_classes, counts)):
        sub_point = gs.center[labels == label,:]
        eigenValueMatrix = generate_graph_from_point(sub_point,radius)

        res_combine_Y, res_combine_U,res_combine_V = color_GFT(gs,labels, label, eigenValueMatrix)
        res_sub_opacity = opacity_GFT(gs,labels, label, eigenValueMatrix)
        res_sub_scale = scale_GFT(gs,labels, label, eigenValueMatrix)
        res_sub_rotate = rotate_GFT(gs,labels, label, eigenValueMatrix)
        res_Y.append(res_combine_Y)
        res_U.append(res_combine_U)
        res_V.append(res_combine_V)
        res_opacity.append(res_sub_opacity)
        res_scale.append(res_sub_scale)
        res_rotation.append(res_sub_rotate)

    return res_Y, res_U, res_V, res_opacity, res_scale, res_rotation
#
# def trans_local(label, count, gs, labels, radius):
#     sub_point = gs.center[labels == label, :]
#     eigenValueMatrix = generate_graph_from_point(sub_point, radius)
#
#     res_combine_Y, res_combine_U, res_combine_V = color_GFT(gs, labels, label, eigenValueMatrix)
#     res_sub_opacity = opacity_GFT(gs, labels, label, eigenValueMatrix)
#     res_sub_scale = scale_GFT(gs, labels, label, eigenValueMatrix)
#     res_sub_rotate = rotate_GFT(gs, labels, label, eigenValueMatrix)
#
#     return label, res_combine_Y, res_combine_U, res_combine_V, res_sub_opacity, res_sub_scale, res_sub_rotate
#
# def graph_trans_p(gs, labels, radius):
#     num_classes, counts = np.unique(labels, return_counts=True)
#
#     res_Y = [None] * len(num_classes)
#     res_U = [None] * len(num_classes)
#     res_V = [None] * len(num_classes)
#     res_opacity = [None] * len(num_classes)
#     res_scale = [None] * len(num_classes)
#     res_rotation = [None] * len(num_classes)
#
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [
#             executor.submit(trans_local, label, count, gs, labels, radius)
#             for label, count in zip(num_classes, counts)
#         ]
#
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(num_classes)):
#             label, res_combine_Y, res_combine_U, res_combine_V, res_sub_opacity, res_sub_scale, res_sub_rotate = future.result()
#             res_Y[int(label)] = res_combine_Y
#             res_U[int(label)] = res_combine_U
#             res_V[int(label)] = res_combine_V
#             res_opacity[int(label)] = res_sub_opacity
#             res_scale[int(label)] = res_sub_scale
#             res_rotation[int(label)] = res_sub_rotate
#
#     return res_Y, res_U, res_V, res_opacity, res_scale, res_rotation