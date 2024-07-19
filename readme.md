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

## Result
you will find the following files after run the script:  
xxx_geo.bin: bitrate for GS center  
xxx_attr.bin: bitrate for GS attributes  
xxx_rec.ply: reconstructed GS sample after compression  
xxx.json: compression parameters  
xxx_time.txt: compression time log  

## GSQA dataset
Google Cloud Drive：https://drive.google.com/drive/folders/1yPYlFsbamrSgkdSAc1J5byozADljrbbp?usp=sharing
Baidu Cloud Drive: https://pan.baidu.com/s/14zJyYmbj8pMlsPqnBMBuzQ password: 6njd   
