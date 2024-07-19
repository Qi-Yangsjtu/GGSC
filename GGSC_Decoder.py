import numpy as np
import argparse
from GS_data import GS
from Encoder import encoder, decoder
import os
import json
from bitstream_utils import *
from scipy.spatial import cKDTree
import time
import platform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--coding_cfg', type=str, default=r'.\example\sofa_R01.json')
    parser.add_argument('--result_root', type=str, default=r'.\example')
    config = parser.parse_args()
    
    ggsc_root = os.path.dirname(__file__)
    os_name = platform.system()
    if os_name == "Windows":
        tmc_path = os.path.join(ggsc_root, "tmc", "windows", "tmc3.exe")
    elif os_name == "Linux":
        tmc_path = os.path.join(ggsc_root, "tmc", "linux", "tmc3")
    elif os_name == "Darwin":
        tmc_path = os.path.join(ggsc_root, "tmc", "mac", "tmc3")
    coding_cfg = load_coding_cfg(config.coding_cfg)
    result_root = config.result_root
    result_name = os.path.basename(config.coding_cfg).split(".")[0]
    print(result_name)
    os.makedirs(result_root, exist_ok=True)
    
    gs_xyz_encoded_bin = coding_cfg.geo_bitstream.path
    xyz_quant_bits = int(coding_cfg.XYZ.quantization_bits)
    qg = coding_cfg.XYZ.position_quantization_scale
    
    start_time = time.time()
    # decoding xyz with GPCC
    gs_xyz_decoded_ply = os.path.join(result_root, "{}_xyz_quant{:02d}_rec.ply".format(result_name, xyz_quant_bits))
    tmc_decode_cmd = "{} --mode=1 --compressedStreamPath={} --reconstructedDataPath={} --outputBinaryPly=0".\
                                        format(tmc_path, gs_xyz_encoded_bin, gs_xyz_decoded_ply)
    print(tmc_decode_cmd)
    os.system(tmc_decode_cmd)
    end_time = time.time()
    gpcc_decoding_time =end_time - start_time
    print("GPCC decoding time: {}".format(gpcc_decoding_time))
    
    # dequant xyz
    gs_xyz_rec = loadXYZFromPLY(gs_xyz_decoded_ply)
    os.remove(gs_xyz_decoded_ply)
    min_xyz = ["x_min", "y_min", "z_min"]
    max_xyz = ["x_max", "y_max", "z_max"]
    gsXYZDequant = np.zeros_like(gs_xyz_rec, dtype=np.float64)
    for i, column in enumerate(gs_xyz_rec.T):
        min_value = getattr(coding_cfg.XYZ, min_xyz[i])
        max_value = getattr(coding_cfg.XYZ, max_xyz[i])
        gsXYZDequant[:, i] = dequantizationFloat(column, min_value, max_value, xyz_quant_bits)
    
    qY = coding_cfg.Y.quantization_bits
    qU = coding_cfg.U.quantization_bits
    qV = coding_cfg.V.quantization_bits
    qO = coding_cfg.opacity.quantization_bits
    qS = coding_cfg.scale.quantization_bits
    qR = coding_cfg.rotation.quantization_bits
    print("AC decoding attributes")
    bitstream_file = coding_cfg.attribute_bitstream.path
    start_time = time.time()
    with open(bitstream_file, "rb") as f:
        bitstream = f.read()
    attr_rec = []
    for attr, quant_bits in zip(["Y", "U", "V", "opacity", "scale", "rotation"], [qY, qU, qV, qO, qS, qR]):
    # for attr, quant_bits in zip(["opacity", "Y", "U", "V", "scale", "rotation"], [qO, qY, qU, qV, qS, qR]):
        print("AC decoding {}".format(attr))
        sub_GS_num = int(getattr(getattr(coding_cfg, attr), "sub_GS_num"))
        data_dim = int(getattr(getattr(coding_cfg, attr), "dimension"))
        quant_bits = int(getattr(getattr(coding_cfg, attr), "quantization_bits"))
        data_attr_min = getattr(getattr(coding_cfg, attr), "min_value")
        data_attr_max = getattr(getattr(coding_cfg, attr), "max_value")
        data_attr_rec, bitstream = ACDecDataP(bitstream, data_dim, quant_bits, sub_GS_num, data_attr_min, data_attr_max)
        attr_rec.append(data_attr_rec)

    end_time = time.time()
    ac_decoding_time = end_time - start_time
    print("AC decoding time: {}".format(ac_decoding_time))
    # generate a empty gs
    gs_rec = GS()
    gs_attr_tmp = np.zeros((gs_xyz_rec.shape[0], 62))
    gs_rec.array_to_attri(gs_attr_tmp)
    gs_rec.center = gsXYZDequant
    
    print("Decoding ...")
    start_time = time.time()
    decoded_gs = decoder(gs_rec, coding_cfg.GFT.kd_split_number, attr_rec[0], attr_rec[1], attr_rec[2], attr_rec[3], attr_rec[4], attr_rec[5])
    end_time = time.time()
    gft_decoding_time = end_time - start_time  
    print("GFT decoding time: {}".format(gft_decoding_time))
    
    time_file = os.path.join(result_root, "{}_time.txt".format(result_name))
    time_info = "{} GPCC decoding time: {} GFT decoding time: {} AC decoding time: {}\n".\
        format(result_name, gpcc_decoding_time, gft_decoding_time, ac_decoding_time)
    with open(time_file, 'a', encoding='utf-8') as file:
        file.write(time_info)

    print("Decode finished, file saving...")
    gs_ply_rec_path = os.path.join(result_root, "{}_rec.ply".\
                            format(result_name))
    decoded_gs.write_3DG_to_PLY_binary(gs_ply_rec_path, 1)
