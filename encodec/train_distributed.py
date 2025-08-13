from torch.distributed import init_process_group, destroy_process_group
import torch
import os
from train import set_args, init_logger, load_config
from baseline_data.iter_dataloader import init_iter_dataset
from losses import total_loss, disc_loss, Metrics, MetricsArgs, LinearWarmupCosineAnnealingLR, WarmupScheduler, ReconstructionLoss
from msstftd import MultiScaleSTFTDiscriminator

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import yaml
import random
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import time
from utils import set_random_seed, print_model_details
import jsonschema
import json
from typing import Tuple, Optional


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_cleanup():
    destroy_process_group()


if __name__ == "__main__":
    args = set_args()
    user_name = os.getlogin()

    checkpoint_path = args.resume_from
    # Load the YAML file
    if os.path.exists(checkpoint_path):
        resume=True
        log_dir = checkpoint_path
        config_dict, config = load_config(filepath=f"{checkpoint_path}/config.yaml", schemapath=None)
    else:
        resume=False  
        config_dict, config = load_config(filepath=f"encodec/params/{args.exp_name}.yaml", schemapath=f"encodec/params/schema.json")
        curr_time = datetime.now().strftime("%Y%m%d")
        curr_minute = datetime.now().strftime("%H%M")
        log_dir = f"{args.log_dir}/{curr_time}_{curr_minute}"
        os.makedirs(log_dir, exist_ok=True)

    # init summarywriter and save config, set random seed, set device
    writer = init_logger(log_dir=log_dir, resume=resume)
    set_random_seed(config.common.seed)
    device = torch.device("cuda")
    if not checkpoint_path:
        #save yaml file to log_dir
        with open(f"{log_dir}/config.yaml", "w") as file:
            yaml.dump(config_dict, file)

    ddp_setup()
    rank = int(os.environ['LOCAL_RANK'])
    
    #train
    # init evaluation metrics logger
    metrics_args = MetricsArgs(device=device, datasets=config.dataset.datasets)
    metrics = Metrics(metrics_args)
    
    # load dataset, split into train and val
    _, train_loader = init_iter_dataset(config=config, type="training", datasets=config.dataset.datasets, pin_memory=True, debug_training=args.debug)
    
    

    torch.cuda.empty_cache()
    
    ddp_cleanup()