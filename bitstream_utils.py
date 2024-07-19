import pickle
import os
import numpy as np
from tqdm import tqdm
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import BinaryPPM
import json
import matplotlib.pyplot as plt
import os
from plyfile import PlyData, PlyElement
from types import SimpleNamespace
import concurrent.futures

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in ns.__dict__.items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(i) for i in ns]
    else:
        return ns

def generate_coding_cfg(config):
    coding_cfg = {
        "geo_bitstream": {
            "path": "tmp",
            "bitstream_length": 0,
            "gpcc_encoding_time": 0
        },
        "attribute_bitstream": {
            "path": "tmp",
            "bitstream_length": 0,
            "gft_encoding_time": 0,
            "AC_encoding_time":0 
        },
        "GFT": {
            "colorRateY" : config.rY,
            "colorRateU" : config.rU,
            "colorRateV" : config.rV,
            "opacityRate" : config.rO,
            "scaleRate" : config.rS,
            "rotationRate" : config.rR,
            "kd_split_number": config.kd_split_number
        },
        "XYZ": {
            "dimension": 3,
            "quantization_bits": config.qXYZ,
            "position_quantization_scale": config.qG,
            "x_max": 255,
            "x_min": 0,
            "y_max": 255,
            "y_min": 0,
            "z_max": 255,
            "z_min": 0,
            "decoded_points": 0,
            "bitstream_length": 0
        },
        "Y": {
            "dimension": 16,
            "quantization_bits": config.qY,
            "max_value": 255,
            "min_value": 0,
            "bitstream_length": 0
        },
        "U": {
            "dimension": 16,
            "quantization_bits": config.qU,
            "max_value": 255,
            "min_value": 0,
            "bitstream_length": 0
        },
        "V": {
            "dimension": 16,
            "quantization_bits": config.qV,
            "max_value": 255,
            "min_value": 0,
            "bitstream_length": 0
        },
        "opacity": {
            "dimension": 1,
            "quantization_bits": config.qO,
            "max_value": 255,
            "min_value": 0,
            "bitstream_length": 0
        },
        "scale": {
            "dimension": 3,
            "quantization_bits": config.qS,
            "max_value": 255,
            "min_value": 0,
            "bitstream_length": 0
        },
        "rotation": {
            "dimension": 4,
            "quantization_bits": config.qR,
            "max_value": 255,
            "min_value": 0,
            "bitstream_length": 0
        }
    }
    coding_cfg = json.loads(json.dumps(coding_cfg), object_hook=lambda d: SimpleNamespace(**d))
    return coding_cfg    

def save_coding_cfg(coding_cfg, coding_cfg_file):
    coding_cfg_dict = namespace_to_dict(coding_cfg)
    with open(coding_cfg_file, 'w') as json_file:
        json.dump(coding_cfg_dict, json_file, indent=4)

def load_coding_cfg(coding_cfg_file):
    with open(coding_cfg_file, 'r') as json_file:
        coding_cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    return coding_cfg

def get_data_attr_range(data_attr):
    min_val = data_attr[0].min()  
    max_val = data_attr[0].max()  
    # for array in data_attr:  
    #     min_val = min(min_val, array.min())  
    #     max_val = max(max_val, array.max())
    for array_index, array in enumerate(data_attr):
        min_val = min(min_val, array.min())  
        max_val = max(max_val, array.max())
    return min_val, max_val

def quantizationFloat(data, min_val, max_val, quant_bits):
    precision = 2**quant_bits
    quant_data = (data - min_val) / (max_val - min_val) * (precision - 1) + 0.5
    quant_data = np.floor(quant_data)
    quant_data = quant_data.astype(int)
    return quant_data

def dequantizationFloat(quant_data, min_val, max_val, quant_bits):
    precision = 2**quant_bits
    quant_data = quant_data.astype(float)
    dequant_data = quant_data / (precision - 1) * (max_val - min_val) + min_val
    return dequant_data

def int_list_to_binary_list(int_list, bit_width):
    binary_list = []
    for num in int_list:
        binary_string = format(num, f'0{bit_width}b')
        binary_list.extend(int(bit) for bit in binary_string)
    return binary_list

def binary_list_to_int_list(binary_list, bit_width):
    int_list = []
    for i in range(0, len(binary_list), bit_width):
        binary_chunk = binary_list[i:i + bit_width]
        binary_string = ''.join(str(bit) for bit in binary_chunk)
        int_list.append(int(binary_string, 2))
    return int_list

def ACDecData(bitstream, data_dim, quant_bits, sub_GS_num, data_attr_min, data_attr_max, oriSampleNumBytes=1, actualValueBytes=3):
    data_attr_rec = []
    
    # extract sub_GS_num bitstreams
    segments = []
    for _ in range(sub_GS_num):
        actual_bits = int.from_bytes(bitstream[:actualValueBytes], byteorder='big')
        actual_bytes = (actual_bits + 7) // 8
        segment_length = actualValueBytes + oriSampleNumBytes + actual_bytes

        segments.append(bitstream[:segment_length])
        bitstream = bitstream[segment_length:]
    
    data_attr_rec = [None] * sub_GS_num  
    for local_idx in tqdm(range(sub_GS_num)):
        byte_array = segments[local_idx]
        actual_bits = int.from_bytes(byte_array[:actualValueBytes], byteorder='big')
        actual_bytes = (actual_bits + 7) // 8
        loPNum = int.from_bytes(byte_array[actualValueBytes:actualValueBytes+oriSampleNumBytes], byteorder='big')
        
        bitList = [0] * (actual_bytes * 8)

        byte_start = actualValueBytes + oriSampleNumBytes
        for i in range(actual_bytes):
            byte_chunk = byte_array[byte_start + i]
            byte_chunk = bin(byte_chunk)[2:].zfill(8)
            bits = [int(bit) for bit in byte_chunk]
            start_pos = i * 8
            end_pos = start_pos + 8
            bitList[start_pos:end_pos] = bits
        bitList = bitList[:actual_bits]

        # Binary PPM model
        model = BinaryPPM(k = 3)
        coder = AECompressor(model)
        data_attr_decoded = coder.decompress(bitList, loPNum*data_dim*quant_bits)
        data_attr_decoded = binary_list_to_int_list(data_attr_decoded, quant_bits)
        data_attr_decoded = np.array(data_attr_decoded)
        # dequantization
        data_attr_local_dequant = dequantizationFloat(data_attr_decoded, data_attr_min, data_attr_max, quant_bits)
        # reshape
        data_attr_local = data_attr_local_dequant.reshape(data_dim, loPNum)
        if data_dim == 1:
            data_attr_local = data_attr_local.squeeze()
        data_attr_rec[local_idx] = data_attr_local

    return data_attr_rec, bitstream


def ACEncData(data_attr, data_dim, quant_bits, sub_GS_num, oriSampleNumBytes=1, actualValueBytes=3):
    data_attr_min, data_attr_max = get_data_attr_range(data_attr)

    bitstream_list = [None] * sub_GS_num

    for local_idx in tqdm(range(sub_GS_num)):
        data_attr_local = data_attr[local_idx]
        # flatten
        data_attr_local = data_attr_local.flatten()
        # local point number
        loPNum = int(data_attr_local.shape[0] / data_dim)
        # quantization
        data_attr_local_quant = quantizationFloat(data_attr_local, data_attr_min, data_attr_max, quant_bits)
        
        data_attr_local_quant = int_list_to_binary_list(data_attr_local_quant, quant_bits)
        # Binary PPM model
        model = BinaryPPM(k = 3)
        coder = AECompressor(model)
        data_attr_compressed = coder.compress(data_attr_local_quant)
        
        # comstruct bitstream
        bitList = data_attr_compressed
        actual_bits = len(bitList)
        extra_bits = len(bitList) % 8
        if extra_bits != 0:
            padding_bits = 8 - extra_bits
            bitList.extend([0] * padding_bits)
        
        new_byte_size = actualValueBytes + oriSampleNumBytes + len(bitList) // 8
        new_byte_array = bytearray(new_byte_size)

        new_byte_array[0:actualValueBytes] = actual_bits.to_bytes(actualValueBytes, byteorder="big")
        new_byte_array[actualValueBytes:actualValueBytes + oriSampleNumBytes] = loPNum.to_bytes(oriSampleNumBytes, byteorder='big')

        for i in range(0, len(bitList), 8):
            byte_chunk = ''.join(map(str, bitList[i:i+8]))
            byte = int(byte_chunk, 2)
            new_byte_array[actualValueBytes + oriSampleNumBytes + i//8] = byte
        bitstream_list[local_idx] = new_byte_array

    bitstreams = b''.join(bitstream_list)
    
    return bitstreams, data_attr_min, data_attr_max

def encode_segment(local_idx, data_attr, data_dim, quant_bits, data_attr_min, data_attr_max, actualValueBytes, oriSampleNumBytes):
    data_attr_local = data_attr[local_idx]
    # flatten
    data_attr_local = data_attr_local.flatten()
    # local point number
    loPNum = int(data_attr_local.shape[0] / data_dim)
    # quantization
    data_attr_local_quant = quantizationFloat(data_attr_local, data_attr_min, data_attr_max, quant_bits)

    data_attr_local_quant = int_list_to_binary_list(data_attr_local_quant, quant_bits)
    # Binary PPM model
    model = BinaryPPM(k=3)
    coder = AECompressor(model)
    data_attr_compressed = coder.compress(data_attr_local_quant)

    # construct bitstream
    bitList = data_attr_compressed
    actual_bits = len(bitList)
    extra_bits = len(bitList) % 8
    if extra_bits != 0:
        padding_bits = 8 - extra_bits
        bitList.extend([0] * padding_bits)

    new_byte_size = actualValueBytes + oriSampleNumBytes + len(bitList) // 8
    new_byte_array = bytearray(new_byte_size)

    new_byte_array[0:actualValueBytes] = actual_bits.to_bytes(actualValueBytes, byteorder="big")
    new_byte_array[actualValueBytes:actualValueBytes + oriSampleNumBytes] = loPNum.to_bytes(oriSampleNumBytes, byteorder='big')

    for i in range(0, len(bitList), 8):
        byte_chunk = ''.join(map(str, bitList[i:i+8]))
        byte = int(byte_chunk, 2)
        new_byte_array[actualValueBytes + oriSampleNumBytes + i // 8] = byte
        
    return local_idx, new_byte_array

def ACEncDataP(data_attr, data_dim, quant_bits, sub_GS_num, oriSampleNumBytes=1, actualValueBytes=3):
    data_attr_min, data_attr_max = get_data_attr_range(data_attr)
    
    bitstream_list = [None] * sub_GS_num

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(encode_segment, local_idx, data_attr, data_dim, quant_bits, data_attr_min, data_attr_max, actualValueBytes, oriSampleNumBytes) for local_idx in range(sub_GS_num)]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=sub_GS_num):
            local_idx, result = future.result()
            bitstream_list[local_idx] = result

    bitstreams = b''.join(bitstream_list)
    
    return bitstreams, data_attr_min, data_attr_max

def decode_segment(local_idx, byte_array, data_dim, quant_bits, data_attr_min, data_attr_max, actualValueBytes, oriSampleNumBytes):
    actual_bits = int.from_bytes(byte_array[:actualValueBytes], byteorder='big')
    actual_bytes = (actual_bits + 7) // 8
    loPNum = int.from_bytes(byte_array[actualValueBytes:actualValueBytes + oriSampleNumBytes], byteorder='big')

    bitList = [0] * (actual_bytes * 8)

    byte_start = actualValueBytes + oriSampleNumBytes
    for i in range(actual_bytes):
        byte_chunk = byte_array[byte_start + i]
        byte_chunk = bin(byte_chunk)[2:].zfill(8)
        bits = [int(bit) for bit in byte_chunk]
        start_pos = i * 8
        end_pos = start_pos + 8
        bitList[start_pos:end_pos] = bits
    bitList = bitList[:actual_bits]

    # Binary PPM model
    model = BinaryPPM(k=3)
    coder = AECompressor(model)
    data_attr_decoded = coder.decompress(bitList, loPNum * data_dim * quant_bits)
    data_attr_decoded = binary_list_to_int_list(data_attr_decoded, quant_bits)
    data_attr_decoded = np.array(data_attr_decoded)
    # dequantization
    data_attr_local_dequant = dequantizationFloat(data_attr_decoded, data_attr_min, data_attr_max, quant_bits)
    # reshape
    data_attr_local = data_attr_local_dequant.reshape(data_dim, loPNum)
    if data_dim == 1:
        data_attr_local = data_attr_local.squeeze()

    return local_idx, data_attr_local

def ACDecDataP(bitstream, data_dim, quant_bits, sub_GS_num, data_attr_min, data_attr_max, oriSampleNumBytes=1, actualValueBytes=3):
    segments = []
    for _ in range(sub_GS_num):
        actual_bits = int.from_bytes(bitstream[:actualValueBytes], byteorder='big')
        actual_bytes = (actual_bits + 7) // 8
        segment_length = actualValueBytes + oriSampleNumBytes + actual_bytes

        segments.append(bitstream[:segment_length])
        bitstream = bitstream[segment_length:]

    data_attr_rec = [None] * sub_GS_num

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(decode_segment, local_idx, segments[local_idx], data_dim, quant_bits, data_attr_min, data_attr_max, actualValueBytes, oriSampleNumBytes)
            for local_idx in range(sub_GS_num)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=sub_GS_num):
            local_idx, result = future.result()
            data_attr_rec[local_idx] = result

    return data_attr_rec, bitstream

def saveXYZToPLY(XYZ, ply_path, binary=False):
    dtype_full = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    
    elements = np.empty(XYZ.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, XYZ))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], text = ~binary).write(ply_path)

def loadXYZFromPLY(ply_path):
    plydata = PlyData.read(ply_path)
    xyz = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    sorted_index = np.lexsort((xyz[:,0], xyz[:,1], xyz[:,2]))
    sorted_xyz = xyz[sorted_index]
    return sorted_xyz

def plotAttrDistribution(data_attr_array, result_root):
    for i, column in enumerate(data_attr_array.T):
        plt.figure() 
        plt.hist(column, bins=100, edgecolor='black', alpha=0.7) 
        plt.title(f'Histogram of feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        
        filename = f'histogram_feature_{i}.png'
        filename = os.path.join(result_root, filename)
        plt.savefig(filename)
        
        plt.close()

