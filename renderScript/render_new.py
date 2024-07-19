#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.camera_utils import JSON_to_camera


def render_sets(pipeline : PipelineParams, config, model_path):
    cameras = JSON_to_camera(config)
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        # gaussians.load_ply(os.path.join(model_path,"point_cloud.ply"))
        model_base = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split(".")[0]
        gaussians.load_ply(model_path)
        
        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_path = model_base
        render_path = os.path.join(render_path, "render")
        render_path = config.result_root
        makedirs(render_path, exist_ok=True)
        for idx, view in enumerate(tqdm(cameras, desc="Rendering progress")):
            # if idx in [0, 25, 28]:
            # if idx>1:
            #     break
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, "{:05d}{}.png".format(idx, config.png_suffix)))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--camera_json", type=str, default="/home/old/yangkaifa/dataset/AVS_sequences/PKU_MVHumans/Dance_Dunhuang_Pair_1080/colmap/train/frame_1/output/cameras.json")
    parser.add_argument("--camera_json", type=str, default="./testData/cameras_rotate.json")
    parser.add_argument("--model", type=str, default="./testData/point_cloud.ply")
    parser.add_argument("--result_root", type=str, default="./testData/render")
    parser.add_argument("--png_suffix", type=str, default="")
    # parser.add_argument("--ref_model", type=str, default="/home/yangkaifa/YQ/GS/model/avs_dance_dunhuang_point_cloud.ply")
    # parser.add_argument("--dis_model", type=str, default="/home/yangkaifa/YQ/GS/model/compression_result/avs_dance_dunhuang_point_cloud_qXYZ14_qG10000_cr1000_or1000_sr1000_rr1000_Y12_U_12_V12_O12_S12_R12.ply")
    args = parser.parse_args()
    args.resolution = 1
    args.sh_degree=3
    args.white_background=False
    args.data_device='cuda'
    # Initialize system state (RNG)
    safe_state(args.quiet)
     
    
    render_sets(pipeline.extract(args), args, args.model)
    # render_sets(pipeline.extract(args), args, args.ref_model)
    # render_sets(pipeline.extract(args), args, args.dis_model)
