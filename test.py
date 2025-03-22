import argparse
import datetime
import os
import random
import time
import numpy as np
import torch

from pysgg.engine.trainer import reduce_loss_dict
from pysgg.modeling.detector import build_detection_model
from pysgg.solver import make_optimizer
from pysgg.utils.checkpoint import DetectronCheckpointer
from pysgg.utils.checkpoint import clip_grad_norm
from pysgg.utils import visualize_graph as vis_graph
from pysgg.utils.comm import synchronize, get_rank
from pysgg.utils.logger import setup_logger, debug_print, TFBoardHandler_LEVEL
from pysgg.utils.metric_logger import MetricLogger
from pysgg.utils.miscellaneous import mkdir
from pysgg.solver.lr_scheduler import WarmupMultiStepLR

from util.dataset import make_data_loader
from util.model import build_detection_model
from util.checkpoint import DetectronCheckpointer
from util.inference import run_test,run_val

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="PyTorch Relation Detection test")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument('--output_dir',type=str,default='checkpoints/sgdet-BGNNPredictor')
    parser.add_argument('--dataset_name',type=str,default='VG_stanford')
    parser.add_argument('--dataset_name_test',type=str,default='VG_stanford_test')
    parser.add_argument('--dataset_name_val',type=str,default='VG_stanford_val')
    parser.add_argument('--img_dir',type=str,default='datasets/vg/stanford_spilt/VG_100k_images')
    parser.add_argument('--glove_dir',type=str,default='datasets/vg/stanford_spilt/glove') 
    parser.add_argument('--roidb_file',type=str,default='datasets/vg/VG-SGG-with-attri.h5')
    parser.add_argument('--dict_file',type=str,default='datasets/vg/VG-SGG-dicts-with-attri.json')
    parser.add_argument('--image_file',type=str,default='datasets/vg/image_data.json')

    parser.add_argument('--train_pre_batch',type=int,default=4)
    parser.add_argument('--test_pre_batch',type=int,default=1)
    parser.add_argument('--max_iter',type=int,default=10000)
    parser.add_argument('--size_divisbility',type=int,default=32)

    parser.add_argument('--base_lr',type=float,default=0.008)
    parser.add_argument('--weight_decay',type=float,default=1.0e-05)
    parser.add_argument('--bias_factor',type=float,default=1.0)
    parser.add_argument('--weight_decay_bias',type=float,default=0.0)
    parser.add_argument('--momentum',type=float,default=0.9)

    parser.add_argument('--pretrained_detection_ckpt',type=str,default='checkpoints/sgdet-BGNNPredictor/(2025-03-14_05)BGNN-3-3(resampling)/model_0027000.pth')
    parser.add_argument('--mode',type=str,default='val')
    parser.add_argument('--debug',type=bool,default = True)
    parser.add_argument('--checkpoint_period',type=int,default=5000)

    #################backbone cfg ############
    parser.add_argument('--res_in_channels',type=int,default=64)
    parser.add_argument('--res_out_channels',type=int,default=256)
    parser.add_argument('--backbone_out_channels',type=int,default=256)
   
    # parser.add_argument('--size_divisbility',type=int,default=32)
    
    #################rpn cfg ############


    #################roi_heads cfg ############
    parser.add_argument('--box_head_num_class',type=int,default=151)
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir,'test')

    data_loader_test = make_data_loader(args,mode=args.mode,is_distributed=False)
    model = build_detection_model(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    args.device = device
    
    output_dir = args.output_dir
    checkpointer = DetectronCheckpointer(args, model, save_dir=output_dir)
    _ = checkpointer.load(args.pretrained_detection_ckpt)
    logger = setup_logger("pysgg", output_dir, get_rank())
    run_val(args,model,data_loader_test,False,logger)


