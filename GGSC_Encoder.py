import numpy as np
import argparse
from GS_data import GS
from Encoder import encoder
import os
from bitstream_utils import *
from scipy.spatial import cKDTree
import time
import platform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref_path', type=str, default=r'.\example\sofa.ply', help="The input gs ply file")
    parser.add_argument('--result_dir', type=str, default=r'.\example', help="Bitstream output path")
    parser.add_argument('--result_name', type=str, default='sofaTest', help="Bitstream output prefix")
    parser.add_argument('--qXYZ', type=int, default='8', help='quantization bits of XYZ for GPCC compression')
    # parser.add_argument('--qXYZ', type=int, default='13', help='quantization bits of XYZ for GPCC compression')
    parser.add_argument('--qG', type=float, default='1', help='positionQuantizationScale parameter of GPCC compression, fix to 1')
    parser.add_argument('--rY', type=float, default='1', help='GFT cut ratio for color Y')
    parser.add_argument('--rU', type=float, default='1', help='GFT cut ratio for color U')
    parser.add_argument('--rV', type=float, default='1', help='GFT cut ratio for color V')
    parser.add_argument('--rO', type=float, default='1', help='GFT cut ratio for opacity')
    parser.add_argument('--rS', type=float, default='1', help='GFT cut ratio for scale') 
    parser.add_argument('--rR', type=float, default='1', help='GFT cut ratio for rotation') 
    parser.add_argument('--qY', type=int, default='9', help='quantization bits of Y for arithmetic coding')
    parser.add_argument('--qU', type=int, default='9', help='quantization bits of U for arithmetic coding')
    parser.add_argument('--qV', type=int, default='9', help='quantization bits of V for arithmetic coding')
    parser.add_argument('--qO', type=int, default='9', help='quantization bits of opacity for arithmetic coding')
    parser.add_argument('--qS', type=int, default='9', help='quantization bits of scale for arithmetic coding')
    parser.add_argument('--qR', type=int, default='9', help='quantization bits of rotation for arithmetic coding')
    parser.add_argument('--kd_split_number', type=int, default='200', help='KD-tree split threshold')

    config = parser.parse_args()
    
    ggsc_root = os.path.dirname(__file__)
    os_name = platform.system()
    if os_name == "Windows":
        tmc_path = os.path.join(ggsc_root, "tmc", "windows", "tmc3.exe")
    elif os_name == "Linux":
        tmc_path = os.path.join(ggsc_root, "tmc", "linux", "tmc3")
    elif os_name == "Darwin":
        tmc_path = os.path.join(ggsc_root, "tmc", "mac", "tmc3")
    ref_file = config.ref_path

    ref_root = os.path.dirname(ref_file)
    result_root = config.result_dir
    os.makedirs(result_root, exist_ok=True)
    result_name = config.result_name
    coding_cfg = generate_coding_cfg(config)

    # read gs and sort by x, y, z
    gs_ori = GS.load_gs_off(ref_file)
    print("Load GS with {} primitives".format(gs_ori.get_3DG_count()))
    

    print("Encoding primitive center (XYZ) with GPCC...")
    gsAttributes = gs_ori.attri_to_array()
    gsXYZ = gsAttributes[:, 0:3]
    gsXYZQuant = np.zeros_like(gsXYZ, dtype=np.int32)
    xyz_quant_bits = coding_cfg.XYZ.quantization_bits
    print("Quantization bits of XYZ: {}".format(xyz_quant_bits))
    min_xyz = ["x_min", "y_min", "z_min"]
    max_xyz = ["x_max", "y_max", "z_max"]
    for i, column in enumerate(gsXYZ.T):
        min_value = np.min(column)
        max_value = np.max(column)
        gsXYZQuant[:, i] = quantizationFloat(column, min_value, max_value, xyz_quant_bits)
        setattr(coding_cfg.XYZ, min_xyz[i], min_value)
        setattr(coding_cfg.XYZ, max_xyz[i], max_value)
    gs_xyz_ply_quant = os.path.join(result_root, result_name + '_xyz_quant{:02d}.ply'.format(xyz_quant_bits))
    saveXYZToPLY(gsXYZQuant, gs_xyz_ply_quant, True)
    
    start_time = time.time()
    # encoding xyz with GPCC
    quant_base_name = os.path.basename(gs_xyz_ply_quant).split('.')[0]
    qg = coding_cfg.XYZ.position_quantization_scale
    # bitrate of GS primitive, named "xxx_geo.bin"
    gs_xyz_encoded_bin = os.path.join(result_root, "{}_geo.bin".format(result_name))
    tmc_encode_cmd = "{} --mode=0 --positionQuantizationScale={} --trisoupNodeSizeLog2=0 --neighbourAvailBoundaryLog2=8 \
            --intra_pred_max_node_size_log2=6 --inferredDirectCodingMode=0 --maxNumQtBtBeforeOt=4\
            --uncompressedDataPath={} --compressedStreamPath={}".format(tmc_path, qg, gs_xyz_ply_quant, gs_xyz_encoded_bin)
    print(tmc_encode_cmd)
    os.system(tmc_encode_cmd)
    end_time = time.time()
    gpcc_encoding_time = end_time-start_time
    print("GPCC encoding time: {}".format(gpcc_encoding_time))
    print("Encoded bitstream length: {}".format(coding_cfg.geo_bitstream.bitstream_length))

    coding_cfg.geo_bitstream.path = gs_xyz_encoded_bin
    coding_cfg.geo_bitstream.bitstream_length = os.path.getsize(gs_xyz_encoded_bin)
    coding_cfg.geo_bitstream.gpcc_encoding_time = gpcc_encoding_time
    # decoding xyz with GPCC
    # decoded GS sample: xxx_rec.ply
    gs_xyz_decoded_ply = os.path.join(result_root, "{}_rec.ply".format(quant_base_name))
    tmc_decode_cmd = "{} --mode=1 --compressedStreamPath={} --reconstructedDataPath={} --outputBinaryPly=0".\
                                        format(tmc_path, gs_xyz_encoded_bin, gs_xyz_decoded_ply)
    print(tmc_decode_cmd)
    os.system(tmc_decode_cmd)


    # dequant decoded XYZ and recosntruct gs
    gs_xyz_rec = loadXYZFromPLY(gs_xyz_decoded_ply)
    os.remove(gs_xyz_ply_quant)
    os.remove(gs_xyz_decoded_ply)
    
    coding_cfg.XYZ.decoded_points = gs_xyz_rec.shape[0]
    gsXYZDequant = np.zeros_like(gs_xyz_rec, dtype=np.float64)
    for i, column in enumerate(gs_xyz_rec.T):
        min_value = getattr(coding_cfg.XYZ, min_xyz[i])
        max_value = getattr(coding_cfg.XYZ, max_xyz[i])
        gsXYZDequant[:, i] = dequantizationFloat(column, min_value, max_value, xyz_quant_bits)
        
    tree = cKDTree(gsXYZ)
    distances, indices = tree.query(gsXYZDequant, k=1)
    attributes = gsAttributes[indices, 3:]
    gsAttributesNew = np.hstack((gsXYZDequant, attributes))
    gs_rec = GS()
    gs_rec.array_to_attri(gsAttributesNew)

    colorRate = SimpleNamespace(Y=1, U=1, V=1)
    colorRate.Y = coding_cfg.GFT.colorRateY
    colorRate.U = coding_cfg.GFT.colorRateU
    colorRate.V = coding_cfg.GFT.colorRateV
    opacityRate = coding_cfg.GFT.opacityRate
    scaleRate = coding_cfg.GFT.scaleRate
    rotationRate = coding_cfg.GFT.rotationRate
    print("Encoding primitive attributes with graph signal processing...")
    start_time = time.time()
    # Saving GFT results when encoding, accelerate decoding
    gft_feature_file = os.path.join(result_root, "gft_feature_qXYZ{:02d}.pkl".format(xyz_quant_bits))

    res_Y, res_U, res_V, res_opacity, res_scale, res_rotation = encoder(gs_rec, gft_feature_file, coding_cfg.GFT.kd_split_number, colorRate, opacityRate, scaleRate, rotationRate)
    end_time = time.time()
    gft_encoding_time = end_time-start_time
    print("GFT frequency clipping time: {}".format(gft_encoding_time))
    coding_cfg.attribute_bitstream.gft_encoding_time = gft_encoding_time
    
    qY = coding_cfg.Y.quantization_bits
    qU = coding_cfg.U.quantization_bits
    qV = coding_cfg.V.quantization_bits
    qO = coding_cfg.opacity.quantization_bits
    qS = coding_cfg.scale.quantization_bits
    qR = coding_cfg.rotation.quantization_bits
    print("AC encoding attributes")
    # bitrate of GS attributes, named "xxx_attr.bin"
    bitstream_file = os.path.join(result_root, "{}_attr.bin".format(result_name))
    start_time = time.time()
    bitstreams = bytearray()
    for attr, quant_bits in zip(["Y", "U", "V", "opacity", "scale", "rotation"], [qY, qU, qV, qO, qS, qR]):
        print("AC encoding {}".format(attr))

        data_attr = eval(f"res_{attr}")
        sub_GS_num = len(data_attr)
        setattr(getattr(coding_cfg, attr), "sub_GS_num", sub_GS_num)
        data_dim = getattr(coding_cfg, attr).dimension
        bitstream, data_attr_min, data_attr_max = ACEncDataP(data_attr, data_dim, quant_bits, sub_GS_num)
        setattr(getattr(coding_cfg, attr), "min_value", data_attr_min)
        setattr(getattr(coding_cfg, attr), "max_value", data_attr_max)
        setattr(getattr(coding_cfg, attr), "bitstream_length", len(bitstream))
        bitstreams.extend(bitstream)

    with open(bitstream_file, 'wb') as f:
        f.write(bitstreams)
    end_time = time.time()
    ac_encoding_time = end_time-start_time
    print("AC Encoding time: {}".format(ac_encoding_time))
    
    coding_cfg.attribute_bitstream.path = bitstream_file
    coding_cfg.attribute_bitstream.bitstream_length = os.path.getsize(bitstream_file)
    coding_cfg.attribute_bitstream.AC_encoding_time = ac_encoding_time
    
    time_file = os.path.join(result_root, "{}_time.txt".format(result_name))
    time_info = "{} GPCC encoding time: {} GFT encoding time: {} AC encoding time: {}\n".format(result_name, gpcc_encoding_time, gft_encoding_time, ac_encoding_time)
    with open(time_file, 'a', encoding='utf-8') as file:
        file.write(time_info)
    
    print("Decode finished, file saving...")
    coding_cfg_file = os.path.join(result_root, "{}.json".format(result_name))
    save_coding_cfg(coding_cfg, coding_cfg_file)