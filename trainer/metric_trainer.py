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
import os
from utils.logger import setup_logger
from layers import make_loss
def create_supervised_trainer(model, optimizer, loss_fn,view_num,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)


    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        fts, target = batch   #target为id
        fts = fts.to(device) if torch.cuda.device_count() >= 1 else fts
        ft_fused,ft_query = model(fts)
        loss,dis_mat = loss_fn(ft_fused,ft_query)
        loss.backward()
        optimizer.step()
        label_in_bath = torch.arange(0,dis_mat.shape[0],1)
        label_in_bath = label_in_bath.to(device) if torch.cuda.device_count() >= 1 else label_in_bath
        acc =(dis_mat.min(1)[1] == label_in_bath).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)

def create_supervised_evaluator(model, metrics, loss_fn, view_num,device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            fts, target = batch  # target为id
            fts = fts.to(device) if torch.cuda.device_count() >= 1 else fts
            target = target[::view_num+1].to(device) if torch.cuda.device_count() >= 1 else target[::view_num+1]
            ft_fused, ft_query = model(fts)

            loss, dis_mat = loss_fn(ft_fused, ft_query)
            label_in_bath = torch.arange(0, dis_mat.shape[0], 1)
            label_in_bath = label_in_bath.to(device) if torch.cuda.device_count() >= 1 else label_in_bath
            acc = (dis_mat.min(1)[1] == label_in_bath).float().mean()
            return ft_fused, ft_query, target, loss.item(), acc.item()

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine

def do_train_val(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_type,
        expirement_name,
        start_epoch,
        view_num

):
    loss_fn = make_loss(cfg, loss_type)
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    if torch.cuda.is_available():
        device = cfg.MODEL.DEVICE
    else:
        device="cpu"

    epochs = cfg.SOLVER.MAX_EPOCHS

    output_dir = cfg.OUTPUT_DIR +"\\" + expirement_name
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger(expirement_name, output_dir, 0)
    logger.info("Start training")

    trainer = create_supervised_trainer(model, optimizer, loss_fn,view_num=view_num, device=device)

    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP( loss_type=loss_type, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},loss_fn=loss_fn, view_num=view_num,device=device)
    checkpointer = ModelCheckpoint(output_dir, expirement_name, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    # for old ignite version
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
    #                                                                  'optimizer': optimizer.state_dict()})
    #for new ignite version,  refer to  https://github.com/pytorch/ignite/issues/529
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})


    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    # 输出的x有两个值，一个是Loss，一个Acc,对应create_supervised_trainer的update函数的输出
    # 每个epoch开始时清空buffer，
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    RunningAverage(output_transform=lambda x: x[3]).attach(evaluator, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[4]).attach(evaluator, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch


    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()


    #每个iteration结束时，记录avg_loss和avg_acc
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = int(timer.step_count)
        logger.info("Iteration[{}] Loss: {:.3f}, Acc: {:.3f}"
            .format(iter,
                    engine.state.output[0], engine.state.output[1]))
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter,
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    # 每个epcoh结束时，调用evaluator
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        logger.info("epoch结束，开始进行评价")
        if engine.state.epoch % eval_period == 0:
        # if True:
            evaluator.run(val_loader)    #只执行一个epock
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_compare_loss(engine):
        iter = engine.state.iteration
        if iter % log_period == 0:
            logger.info("Evaluator----- Iteration[{}] Loss: {:.3f}, Acc: {:.3f}"
                        .format(iter, engine.state.output[3], engine.state.output[4]))

    # print('开始进行评价')
    # evaluator.run(val_loader)  # 只执行一个epock
    # cmc, mAP = evaluator.state.metrics['r1_mAP']
    # print(cmc,mAP)

    trainer.run(train_loader, max_epochs=epochs)

