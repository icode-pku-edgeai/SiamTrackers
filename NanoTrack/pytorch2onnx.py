# Copyright (c) SenseTime. All Rights Reserved.
import argparse
import os
import torch
import sys 

sys.path.append(os.getcwd())

from nanotrack.core.config import cfg

from nanotrack.utils.model_load import load_pretrain
from nanotrack.models.model_builder import ModelBuilder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='lighttrack')

parser.add_argument('--config', type=str, default='./models/config/configv3.yaml',help='config file')

parser.add_argument('--snapshot', default='models/snapshot/coco_tank_allDatasets.pth', type=str,  help='snapshot models to eval')

args = parser.parse_args()

def main():

    cfg.merge_from_file(args.config)

    model = ModelBuilder()

    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

    model = ModelBuilder()

    model = load_pretrain(model, args.snapshot)
    
    model.eval().to(device)

    backbone_net = model.backbone

    head_net = model.ban_head
    
    # backbone input-xf
    backbone_x = torch.randn([1, 3, 127, 127], device=device)
    export_onnx_file_path= './models/onnx/backbone127.onnx'
    torch.onnx.export(backbone_net, backbone_x, export_onnx_file_path, input_names=['Tinput'], output_names=['Toutput'], verbose=True, opset_version=14)   
    
    backbone_x = torch.randn([1, 3, 255, 255], device=device)
    export_onnx_file_path= './models/onnx/backbone255.onnx'
    torch.onnx.export(backbone_net, backbone_x, export_onnx_file_path, input_names=['Tinput'], output_names=['Toutput'], verbose=True, opset_version=14)   


    # head  change forward  /media/dell/Data/NanoTrack/nanotrack/models/model_builder.py
    head_zf, head_xf = torch.randn([1, 96, 8, 8],device=device), torch.randn([1, 96, 16, 16],device=device)
    export_onnx_file_path= './models/onnx/head.onnx'
    torch.onnx.export(head_net,(head_zf,head_xf), export_onnx_file_path, input_names=['input1','input2'], output_names=['output1','output2'],verbose=True,opset_version=14) 
    
    # # 模型简化,否则onnx转换成ncnn会报错
    
    # # """
    # # 命令行： python3 -m  onnxsim   input_your_mode_name  output_onnx_model
    # # github: github.com/daquexian/onnx-simplifier
    # # """s
    import onnx
    from onnxsim  import simplify # if no module named 'onnxsim' , you should run pip install onnx-simplifier  in  terminal
   
    filename =  './models/onnx/backbone127.onnx'  
    simplified_model,check =simplify('./models/onnx/backbone127.onnx',skip_fuse_bn=False) 
    onnx.save_model(simplified_model,filename)

    filename =  './models/onnx/backbone255.onnx'  
    simplified_model,check =simplify('./models/onnx/backbone255.onnx',skip_fuse_bn=False) 
    onnx.save_model(simplified_model,filename)

    filename =  './models/onnx/head.onnx'   
    simplified_model,check =simplify('./models/onnx/head.onnx',skip_fuse_bn=False)
    onnx.save_model(simplified_model,filename)  

if __name__ == '__main__':
    main() 
