# encoding: utf-8
"""
@author:  shuai dong
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
import random
from utils.reid_metric import R1_mAP
import cv2
import numpy as np
from layers import make_VPM_loss_val