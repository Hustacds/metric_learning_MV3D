# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""
# import numpy as np
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torchvision.models as models
# import cv2
# from models import FFM
# import torch.optim as optim
# import random
# from torch.utils.data import Dataset,DataLoader,TensorDataset
# from ignite.engine import Engine, Events
# from ignite.handlers import ModelCheckpoint, Timer
# from ignite.metrics import RunningAverage


import argparse
import os
import sys
# import torch

from torch.backends import cudnn
sys.path.append('.')
from config import cfg
from utils.logger import setup_logger
from trainer.metric_trainer import do_train_val
from models.FFM import FFM
from data.build_dataloader import make_data_loader
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR
import torch.optim as optim
def compare(cfg,logger):

    feature_dataset = cfg.DATASETS.FEATURESET_NAMES

    net_type_list = cfg.EXPERIMENT.NET_TYPE
    loss_type_list = cfg.EXPERIMENT.LOSS_TYPE
    view_num_list = cfg.EXPERIMENT.VIEW_NUM
    # print(net_type_list)
    # print(loss_type_list)
    # print(view_num_list)


    l_dataset_combine = len(feature_dataset)
    #循环1，不同的数据集组合
    for i_dataset_group in range(l_dataset_combine):
        feature_group = feature_dataset[i_dataset_group]

        #循环2，不同的视图数量
        for view_num in view_num_list:
            #循环3，不同特征融合方式
            for net_type in net_type_list:
                #循环4，不同的距离度量方式
                for loss_type in loss_type_list:
                    # 定义dataloader
                    train_loader, val_loader,train_id_nums,val_id_nums = make_data_loader(cfg,feature_group,view_num)
                    experiment_name = feature_group+'__view_'+str(view_num)+'__'+net_type+'__' +loss_type
                    # print(experiment_name)
                    logger.info("开始新的训练与测试:{}".format(experiment_name))

                    loss_fn = make_loss(cfg)

                    #构建网络
                    model = FFM(net_type,loss_type,view_num,train_id_nums)
                    # optimizer = make_optimizer(cfg, model)
                    start_epoch = 0
                    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                    #                               cfg.SOLVER.WARMUP_FACTOR,
                    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)

                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40, 50], 0.1)

                    #训练评估

                    do_train_val(cfg,
                            model,
                            train_loader,
                            val_loader,
                            optimizer,
                            scheduler,
                            loss_fn,
                            # num_query,
                            experiment_name,
                            start_epoch
                    )
    return 0


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("metric_learning_3D", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    compare(cfg,logger)

if __name__ == '__main__':
    main()