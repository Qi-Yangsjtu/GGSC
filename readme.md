# A Benchmark for Gaussian Splatting Compression and Quality Assessment Study

## Sample preparation
Download example from https://drive.google.com/file/d/1QYOBdPcS0M2YXMbV4G_N7-_kKy9htpaw/view?usp=sharing  
unzip example.zip and move the folder to the same path of this project  

## Python environment
pip install arithmetic_compressor  
pip install scipy  
pip install matplotlib  

## Run script
check script "test_example.py"  
set necessary file path and run

Note: our "renderScript/render_new.py" uses a function "JSON_to_camera", please use the provided "renderScript/camera_utils.py" to replace original 3DGS "camera_utils.py", or copy the function "JSON_to_camera" into your own "camera_utils.py"

## Result
you will find the following files after run the script:  
xxx_geo.bin: bitrate for GS center  
xxx_attr.bin: bitrate for GS attributes  
xxx_rec.ply: reconstructed GS sample after compression  
xxx.json: compression parameters  
xxx_time.txt: compression time log  

## GSQA dataset
OneDrive：https://1drv.ms/f/c/3dbe3858aa085846/EjN7TaKq5QBNhgE83zV5plIB5RcFIUmlV0VCvzsXpFp8pQ?e=5aGyg8 
Baidu Cloud Drive: https://pan.baidu.com/s/14zJyYmbj8pMlsPqnBMBuzQ password: 6njd   

If you use our code and dataset, please cite our paper；

@inproceedings{yang2024GGSC,
author = {Yang, Qi and Yang, Kaifa and Xing, Yuke and Xu, Yiling and Li, Zhu},
title = {A Benchmark for Gaussian Splatting Compression and Quality Assessment Study},
year = {2024},
isbn = {9798400712739},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3696409.3700172},
doi = {10.1145/3696409.3700172},
booktitle = {Proceedings of the 6th ACM International Conference on Multimedia in Asia},
articleno = {12},
numpages = {8},
keywords = {3D Gaussian Splatting, Compression, Quality Assessment},
series = {MMAsia '24}
}
