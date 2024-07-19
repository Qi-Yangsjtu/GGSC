import os

ggsc_root = os.path.dirname(__file__)
encoder_path = os.path.join(ggsc_root, "GGSC_Encoder.py")
decoder_path = os.path.join(ggsc_root, "GGSC_Decoder.py")

# Path of input GS sample
test_gs = os.path.join(ggsc_root, "example", "sofa.ply")
# Path of camera file for the input GS, for rendering
camera_json = os.path.join(ggsc_root, "example", "cameras.json")
# Path of result
test_result_root = os.path.join(ggsc_root, "example2")
# Name of result
test_result_name = "sofa_R01"

# Compression parameters
disortionTypes = ["qXYZ", "rY", "rU", "rV", "rO", "rS", "rR", "qY", "qU", "qV", "qO", "qS", "qR"]
distortionParas = [9, 0.4, 0.4, 0.4, 0.4, 0.85, 0.4, 6, 6, 6, 6, 8, 6]

# Sample compression
test_encode_cmd = "python {} --ref_path={} --result_dir={} --result_name={} --qXYZ={} --rY={} --rU={} --rV={}\
            --rO={} --rS={} --rR={} --qY={} --qU={} --qV={} --qO={} --qS={} --qR={}".format(
            encoder_path, test_gs, test_result_root, test_result_name, distortionParas[0], distortionParas[1],
            distortionParas[2], distortionParas[3], distortionParas[4], distortionParas[5], distortionParas[6], 
            distortionParas[7], distortionParas[8], distortionParas[9], distortionParas[10], distortionParas[11], 
            distortionParas[12])
test_result_json = os.path.join(test_result_root, "{}.json".format(test_result_name))
test_result_ply = os.path.join(test_result_root, "{}_rec.ply".format(test_result_name))
if not os.path.exists(test_result_json):
    print(test_encode_cmd)
    os.system(test_encode_cmd)

# Sample decoding
test_decode_cmd = "python {} --coding_cfg={} --result_root={}".format(decoder_path, test_result_json, test_result_root)
if not os.path.exists(test_result_ply):
    print(test_decode_cmd)
    os.system(test_decode_cmd)


# Rendering Sample, you need to install gaussian-splatting first,
# and copy render_new.py to the path of gaussian-splatting

# render_script = r"D:\Projects\PythonProjects\gaussian-splatting\render_new.py"
# compression_render_root = os.path.join(ggsc_root, "example", "render")
# render_cmd = "python {} --camera_json={} --model={} --result_root={} --png_suffix={} --white_background".\
#             format(render_script, camera_json, test_result_ply, compression_render_root, "sofa_R01")
# print(render_cmd)
# os.system(render_cmd)

