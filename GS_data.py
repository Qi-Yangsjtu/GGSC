import numpy as np
from plyfile import PlyData, PlyElement
from sup_util import get_decimal, GS_sort
import os
class GS:
    def __init__(self, center=None, normal=None, baseColor=None, R_SH=None, G_SH=None, B_SH=None, opacity=None, scale=None, rotate=None):
        self.center = center
        self.normal = normal
        self.baseColor = baseColor
        self.R_SH = R_SH
        self.G_SH = G_SH
        self.B_SH = B_SH
        self.opacity = opacity
        self.scale = scale
        self.rotate = rotate


    @classmethod
    def load_gs_off(cls, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])

        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])


        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        center = xyz
        baseColor = features_dc
        ss = int(features_extra.shape[1]/3)
        R_SH = features_extra[:, 0:ss]
        G_SH = features_extra[:, ss:2*ss]
        B_SH = features_extra[:, 2*ss:]

        opacity = opacities
        scale = scales
        rotate =rots
        normal = np.zeros((xyz.shape[0], 3))

        center, normal, baseColor, R_SH, G_SH, B_SH, opacity, scale, rotate = \
            GS_sort(center, normal, baseColor, R_SH, G_SH, B_SH, opacity, scale, rotate)

        return cls(center, normal, baseColor, R_SH, G_SH, B_SH, opacity, scale, rotate)

    @classmethod
    def load_xyz_only(cls, xyz_path, dec_num):
        gs_rec = PlyData.read(xyz_path)
        xyz = np.stack((np.asarray(gs_rec.elements[0]["x"]),
                        np.asarray(gs_rec.elements[0]["y"]),
                        np.asarray(gs_rec.elements[0]["z"])), axis=1)
        xyz = xyz.astype('int')
        if not(dec_num == None):
            xyz = xyz / (10 ** dec_num)
        sorted_index = np.lexsort((xyz[:,0], xyz[:,1], xyz[:,2]))
        sorted_xyz = xyz[sorted_index]
        normal = np.zeros((xyz.shape[0], 3))
        baseColor = np.zeros((xyz.shape[0], 3))
        R_SH = np.zeros((xyz.shape[0], 15))
        G_SH = np.zeros((xyz.shape[0], 15))
        B_SH = np.zeros((xyz.shape[0], 15))
        opacity = np.zeros((xyz.shape[0]))
        scale = np.zeros((xyz.shape[0], 3))
        rotate = np.zeros((xyz.shape[0], 4))
        return cls(sorted_xyz, normal, baseColor, R_SH, G_SH, B_SH, opacity, scale, rotate)

    def write_point_cloud_geometry_only(self, filedir, integer = 0):
        if os.path.exists(filedir):
            os.system('rm '+ filedir)
        f = open(filedir, 'a+')
        f.writelines(['ply\n', 'format ascii 1.0\n'])
        f.write('element vertex ' + str(self.center.shape[0]) + '\n')
        f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
        f.write('end_header\n')
        coords = self.center.astype('float')
        if integer == 0:
            for p in coords:
                f.writelines([str(p[0]), ' ', str(p[1]), ' ', str(p[2]), '\n'])
            f.close()
        else:
            rate = self.center_decimal_places()
            multipler = 10**rate
            for p in coords:
                f.writelines([str(int(p[0]*multipler)), ' ', str(int(p[1]*multipler)), ' ', str(int(p[2]*multipler)), '\n'])
            f.close()
    
    def write_point_cloud_with_color(self, filedir):
        if os.path.exists(filedir):
            os.system('rm '+ filedir)
        f = open(filedir, 'a+')
        f.writelines(['ply\n', 'format ascii 1.0\n'])
        f.write('element vertex ' + str(self.center.shape[0]) + '\n')
        f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
        f.writelines(['property uchar red\n', 'property uchar green\n', 'property uchar blue\n'])
        f.write('end_header\n')
        coords = self.center.astype('float')
        color = (self.get_3DG_baseRGB_clip() * 255).astype('uint8')
        for i in range(len(color)):
            p = coords[i]
            j = color[i]
            f.writelines([str(p[0]), ' ', str(p[1]), ' ', str(p[2]), ' ', str(j[0]), ' ', str(j[1]), ' ', str(j[2]),'\n'])
        f.close()
        return
    
    def get_dtype_for_quant_bits(self, quant_bits, signed=False):
        if signed:
            if quant_bits <= 8:
                return 'i1'
            elif quant_bits <= 16:
                return 'i2'
            elif quant_bits <= 32:
                return 'i4'
            else:
                return 'i8'
        else:
            if quant_bits <= 8:
                return 'u1'
            elif quant_bits <= 16:
                return 'u2'
            elif quant_bits <= 32:
                return 'u4'
            else:
                return 'u8'
    
    def write_point_cloud_int(self, ply_path, quant_bits, geometry_only, binary=False, signed=False):
        attr_array = self.attri_to_array()
        
        attr_dtype = self.get_dtype_for_quant_bits(quant_bits, signed)
        attr_dtype = 'f4'
        dtype_full = [(attribute, attr_dtype) for attribute in self.construct_list_of_attributes()]
        # dtype_full = dtype_full[:3] + dtype_full[6:]
        if geometry_only:
            dtype_full = dtype_full[:3]
            attr_array = attr_array[:, :3]
        
        elements = np.empty(attr_array.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attr_array))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], text = ~binary).write(ply_path)
        
    
    def construct_list_of_attributes(self):
        data = np.hstack((self.R_SH, self.G_SH, self.B_SH))
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.baseColor.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(data.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotate.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def write_3DG_to_PLY_binary(self, filedir, flag_binary):
        # 1 for binary, 0 for ASCII
        if os.path.exists(filedir):
            os.system('rm '+ filedir)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        xyz = self.center

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        data = np.hstack((self.center, self.normal, self.baseColor, self.R_SH, self.G_SH, self.B_SH))
        data = np.hstack((data, np.expand_dims(self.opacity, axis=1)))
        attributes = np.hstack((data, self.scale, self.rotate))

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        if flag_binary == 1:
            PlyData([el]).write(filedir)
        else:
            PlyData([el], text = True).write(filedir)

    def get_3DG_count(self):
        return len(self.center)

    def get_3GD_bbox(self, flag):
        point_cloud = self.center
        x_max, x_min = np.max(point_cloud[:, 0]), np.min(point_cloud[:, 0])
        y_max, y_min = np.max(point_cloud[:, 1]), np.min(point_cloud[:, 1])
        z_max, z_min = np.max(point_cloud[:, 2]), np.min(point_cloud[:, 2])

        x_dim = x_max - x_min
        y_dim = y_max - y_min
        z_dim = z_max - z_min
        if flag == 0:
            return x_dim, y_dim, z_dim
        else:
            return x_max, x_min, y_max, y_min, z_max, z_min
    def get_SH_degree(self):
        R_SH = self.R_SH
        SH_drgree = R_SH.shape[1]
        print(SH_drgree)
        if SH_drgree == 15:
            degree = 3
        elif SH_drgree == 8:
            degree = 2
        elif SH_drgree == 3:
            degree = 1
        elif SH_drgree == 0:
            degree = 0
        else:
            degree = -1
        return degree
    def get_3DG_baseRGB_clip(self):
        baseColor = self.baseColor
        SH_C0 = 0.28209479177387814
        color = SH_C0 * baseColor + 0.5
        color[color<0]=0
        return color
    def get_3DG_baseRGB(self):
        baseColor = self.baseColor
        SH_C0 = 0.28209479177387814
        color = SH_C0 * baseColor + 0.5
        return color

    def baseRGB_recover(self, color):
        SH_C0 = 0.28209479177387814
        baseColor = (color - 0.5)/SH_C0
        return baseColor

    def baseRGB_To_baseYUV(self):
        baseRGB = self.get_3DG_baseRGB()
        baseY = 0.299 * baseRGB[:,0] + 0.587 * baseRGB[:,1] + 0.114 * baseRGB[:,2]
        baseU = -0.169 * baseRGB[:,0] - 0.331 * baseRGB[:,1] + 0.500 * baseRGB[:,2]
        baseV = 0.500 * baseRGB[:,0] - 0.419 * baseRGB[:,1] - 0.081 * baseRGB[:,2]
        return baseY, baseU, baseV

    def baseYUV_To_baseRGB(self, baseY, baseU, baseV):

        baseR = baseY + 1.403 * baseV
        baseG = baseY - 0.344 * baseU - 0.714 * baseV
        baseB = baseY + 1.770 * baseU
        return baseR, baseG, baseB

    def SH_RGB_To_SH_YUV(self):
        R_SH = self.R_SH
        G_SH = self.G_SH
        B_SH = self.B_SH

        Y_SH = 0.299 * R_SH + 0.587 * G_SH + 0.114 * B_SH
        U_SH = -0.169 * R_SH - 0.331 * G_SH + 0.500 * B_SH
        V_SH = 0.500 * R_SH - 0.419 * G_SH - 0.081 * B_SH
        return Y_SH, U_SH, V_SH

    def SH_YUV_To_SH_RGB(self, Y_SH, U_SH, V_SH):

        R_SH = Y_SH + 1.403 * V_SH
        G_SH = Y_SH - 0.344 * U_SH - 0.714 * V_SH
        B_SH = Y_SH + 1.770 * U_SH
        return R_SH, G_SH, B_SH

    def center_decimal_places(self):
        coord1 = self.center[0][0]
        coord2 = self.center[0][1]
        coord3 = self.center[0][2]
        len1 = get_decimal(coord1)
        len2 = get_decimal(coord2)
        len3 = get_decimal(coord3)
        return max(len1, len2, len3)

    def attri_to_array(self):
        gsAttri_array = np.hstack((self.center, self.normal, self.baseColor, self.R_SH, self.G_SH, self.B_SH, np.expand_dims(self.opacity, axis=1), self.scale, self.rotate))
        return gsAttri_array
    
    def array_to_attri(self, gsAttri_array):
        self.center = gsAttri_array[:, 0:3]
        self.normal = gsAttri_array[:, 3:6]
        self.baseColor = gsAttri_array[:, 6:9]
        self.R_SH = gsAttri_array[:, 9:24]
        self.G_SH = gsAttri_array[:, 24:39]
        self.B_SH = gsAttri_array[:, 39:54]
        self.opacity = gsAttri_array[:, 54]
        self.scale = gsAttri_array[:, 55:58]
        self.rotate = gsAttri_array[:, 58:62]
