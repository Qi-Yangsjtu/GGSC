import numpy as np
from sup_util import split_point_cloud
from GFT_util import generate_graph_from_point, graph_trans
from tqdm import tqdm
from GS_data import GS
import pickle
import os

# SH clipping
def quantization_color(res_Y, res_U, res_V, labels,rate):
    num_classes, counts = np.unique(labels, return_counts=True)
    i=0
    for label, count in zip(num_classes, counts):
        sub_res_Y = res_Y[i]
        sub_res_U = res_U[i]
        sub_res_V = res_V[i]

        clip_Y = max(int (sub_res_Y.shape[1] * rate.Y), 1)
        clip_U = max(int (sub_res_U.shape[1] * rate.U), 1)
        clip_V = max(int (sub_res_V.shape[1] * rate.V), 1)
        res_Y[i] = sub_res_Y[:, 0:clip_Y]
        res_U[i] = sub_res_U[:, 0:clip_U]
        res_V[i] = sub_res_V[:, 0:clip_V]
        if res_Y[i].shape[1]==0:
            print(i)
        i+=1

# opacity clipping
def quantization_opacity(res_opacity, labels, rate):
    num_classes, counts = np.unique(labels, return_counts=True)
    i=0
    for label, count in zip(num_classes, counts):
        sub_res_opacity = res_opacity[i]
        clip = max(int(sub_res_opacity.shape[0] * rate), 1)
        res_opacity[i] = sub_res_opacity[0:clip]
        i+=1

# scale clipping
def quantization_scale(res_scale, labels, rate):
    num_classes, counts = np.unique(labels, return_counts=True)
    i=0
    for label, count in zip(num_classes, counts):
        sub_res_scale = res_scale[i]
        clip = max(int (sub_res_scale.shape[1] * rate), 1)
        res_scale[i] = sub_res_scale[:, 0:clip]
        i+=1

# rotate clipping
def quantization_rotate(res_rotate, labels, rate):
    num_classes, counts = np.unique(labels, return_counts=True)
    i=0
    for label, count in zip(num_classes, counts):
        sub_res_rotate = res_rotate[i]
        clip = max(int (sub_res_rotate.shape[1] * rate), 1)
        res_rotate[i] = sub_res_rotate[:, 0:clip]
        i+=1
def encoder(gs, gft_features_file, min_sample, colorRate, opacityRate, scaleRate, rotationRate):
    if os.path.exists(gft_features_file):
        print("GFT feature exists, loading")
        with open(gft_features_file, 'rb') as f:
            data = pickle.load(f)
        labels = data['labels']
        res_Y = data['res_Y']
        res_U = data['res_U']
        res_V = data['res_V']
        res_opacity = data['res_opacity']
        res_scale = data['res_scale']
        res_rotation = data['res_rotation']
    else:
        print("Spliting GS (encoding preprocessing)...")
        labels = split_point_cloud(gs, min_sample)
        print("Graph constructing...")

        x,y,z = gs.get_3GD_bbox(0)
        min_dim = min(x,y,z)
        max_dis = min_dim/20
        res_Y, res_U, res_V, res_opacity, res_scale, res_rotation  = graph_trans(gs, labels, max_dis)

        
    print("Quantization ...")
    quantization_color(res_Y, res_U, res_V, labels, colorRate)
    quantization_opacity(res_opacity, labels, opacityRate)
    quantization_scale(res_scale, labels, scaleRate)
    quantization_rotate(res_rotation, labels, rotationRate)
    return res_Y, res_U, res_V, res_opacity, res_scale, res_rotation


def recover_color(res_Y, res_U, res_V, i, num_points, inv_eig, labels, label,
                  new_base_color, new_Y_SH, new_U_SH, new_V_SH):
    sub_res_Y = res_Y[i]
    re_sub_res_Y = np.zeros((sub_res_Y.shape[0], num_points))
    re_sub_res_Y[:, :sub_res_Y.shape[1]] = sub_res_Y
    sub_res_U = res_U[i]
    re_sub_res_U = np.zeros((sub_res_U.shape[0], num_points))
    re_sub_res_U[:, :sub_res_U.shape[1]] = sub_res_U
    sub_res_V = res_V[i]
    re_sub_res_V = np.zeros((sub_res_V.shape[0], num_points))
    re_sub_res_V[:, :sub_res_V.shape[1]] = sub_res_V

    re_sub_Y = re_sub_res_Y @ inv_eig
    re_sub_U = re_sub_res_U @ inv_eig
    re_sub_V = re_sub_res_V @ inv_eig

    new_base_color[labels == label, 0] = re_sub_Y[0, :]
    new_base_color[labels == label, 1] = re_sub_U[0, :]
    new_base_color[labels == label, 2] = re_sub_V[0, :]
    new_Y_SH[labels == label, :] = np.transpose(re_sub_Y[1:, :])
    new_U_SH[labels == label, :] = np.transpose(re_sub_U[1:, :])
    new_V_SH[labels == label, :] = np.transpose(re_sub_V[1:, :])

def recover_opacity(res_opacity, i, num_points, inv_eig, labels, label, new_opacity):
    sub_res_opacity = res_opacity[i]    
    re_sub_res_opacity = np.zeros(num_points)
    if np.isscalar(sub_res_opacity) or (isinstance(sub_res_opacity, np.ndarray) and sub_res_opacity.ndim == 0):
        # If it's a scalar, assign it to the first element of re_sub_res_opacity
        re_sub_res_opacity[0] = sub_res_opacity
    else:
        # If it's an array, assign its values to the beginning of re_sub_res_opacity
        re_sub_res_opacity[0:sub_res_opacity.shape[0]] = sub_res_opacity

    re_sub_opacity = re_sub_res_opacity@inv_eig
    new_opacity[labels==label] = re_sub_opacity

def recover_3DGShape(res_attribute, i, num_points, inv_eig, labels, label, new_attribute):
    sub_res_attribute = res_attribute[i]

    re_sub_res_attribute = np.zeros((sub_res_attribute.shape[0], num_points))
    re_sub_res_attribute[:, :sub_res_attribute.shape[1]] = sub_res_attribute

    re_sub_attribute = re_sub_res_attribute@inv_eig
    new_attribute[labels==label,:] = np.transpose(re_sub_attribute)


def decoder(gs, min_sample, res_Y, res_U, res_V, res_opacity, res_scale, res_rotation):
    print("Spliting GS (decoding preprocessing)...")
    x,y,z = gs.get_3GD_bbox(0)

    min_dim = min(x,y,z)
    max_dis = min_dim/20

    labels = split_point_cloud(gs, min_sample)
    num_classes, counts = np.unique(labels, return_counts=True)
    new_base_color = np.zeros(gs.baseColor.shape)
    new_Y_SH = np.zeros(gs.R_SH.shape)
    new_U_SH = np.zeros(gs.R_SH.shape)
    new_V_SH = np.zeros(gs.R_SH.shape)
    new_opacity = np.zeros(gs.opacity.shape)
    new_scale = np.zeros(gs.scale.shape)
    new_rotation = np.zeros(gs.rotate.shape)

    i=0
    for label, count in tqdm(zip(num_classes, counts)):
        sub_point = gs.center[labels == label, :]
        num_points = sub_point.shape[0]

        eigenvectors = generate_graph_from_point(sub_point, max_dis)
        inv_eig = np.linalg.inv(eigenvectors)
        recover_color(res_Y, res_U, res_V, i, num_points, inv_eig, labels, label,
                      new_base_color, new_Y_SH, new_U_SH, new_V_SH)
        recover_opacity(res_opacity, i, num_points, inv_eig, labels, label, new_opacity)
        recover_3DGShape(res_scale, i, num_points, inv_eig, labels, label, new_scale)
        recover_3DGShape(res_rotation, i, num_points, inv_eig, labels, label, new_rotation)
        i+=1

    new_baseR, new_baseG, new_baseB = gs.baseYUV_To_baseRGB(new_base_color[:,0], new_base_color[:,1],new_base_color[:,2])
    new_R_SH, new_G_SH, new_B_SH = gs.SH_YUV_To_SH_RGB(new_Y_SH, new_U_SH, new_V_SH)


    decode_BaseColor = np.transpose(np.vstack((new_baseR, new_baseG, new_baseB)))
    decode_BaseColor_re = gs.baseRGB_recover(decode_BaseColor)

    decode_gs = GS(gs.center, gs.normal, decode_BaseColor_re, new_R_SH, new_G_SH, new_B_SH, new_opacity, new_scale, new_rotation)


    return decode_gs