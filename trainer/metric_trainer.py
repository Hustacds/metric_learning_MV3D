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

def create_supervised_trainer(model, optimizer, loss_fn,
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
        img, target,imgs_partial,region_label = batch   #target为id

        # img_eg = imgs_partial.cpu().numpy()[0]
        # img_eg = np.transpose(img_eg, (1, 2, 0))
        # img_eg = cv2.cvtColor(img_eg, cv2.COLOR_BGR2RGB)
        # print('trainer 中dataloader 输入的图像尺寸为{}'.format(img_eg.shape))
        # cv2.imshow('input', img_eg)
        # cv2.waitKey(1)

        imgs_partial = imgs_partial.to(device) if torch.cuda.device_count() >= 1 else imgs_partial
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        region_label = region_label.to(device) if torch.cuda.device_count() >= 1 else region_label

        feature_map, probability_map, visibility_score, feature_region, class_socre = model(imgs_partial)
        # print(feature_map.shape)
        # print(probability_map.shape)
        # print(visibility_score.shape)
        # print(feature_region.shape)
        # print(class_socre.shape)
        loss,score,l_r,l_id,l_tri = loss_fn(feature_map, probability_map, visibility_score, feature_region, class_socre, target, region_label)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item(),l_r.item(),l_id.item(),l_tri.item()

    return Engine(_update)

#在eval中，只生成feature，而不考虑id判断是否正确了，这是由于测试集中的Id和训练集中的不一样，需要直接去判断。
def create_supervised_evaluator(model, metrics, loss_fn_val, device=None):
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
            img, pids, camids,imgs_partial,region_label = batch
            imgs_partial = imgs_partial.to(device) if torch.cuda.device_count() >= 1 else imgs_partial

            target = torch.tensor(pids, dtype=torch.int64)
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            region_label = region_label.to(device) if torch.cuda.device_count() >= 1 else region_label
            feature_map, probability_map, visibility_score, feature_region, class_socre = model(imgs_partial)
            loss,  l_r,  l_tri = loss_fn_val( probability_map, feature_region,
                                                     target, region_label)
            # print("evaluator===========",pids)
            return feature_region, pids, camids,visibility_score, loss.item(), l_r.item(),l_tri.item()

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
        loss_fn,
        # num_query,
        expirement_name,
        start_epoch

):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR

    if torch.cuda.is_available():
        device = cfg.MODEL.DEVICE
    else:
        device="cpu"

    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")


    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

    num_query = 10

    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},loss_fn_val=loss_fn, device=device)
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
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'avg_l_r')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'avg_l_id')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'avg_l_tri')

    RunningAverage(output_transform=lambda x: x[4]).attach(evaluator, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[5]).attach(evaluator, 'avg_l_r')
    RunningAverage(output_transform=lambda x: x[6]).attach(evaluator, 'avg_l_tri')

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
        logger.info("Iteration[{}] Loss: {:.3f}, Acc: {:.3f}, L_r:{:.3f}, L_id:{:.3f}, L_tri:{:.3f}"
            .format(iter,
                    engine.state.output[0], engine.state.output[1],
                    engine.state.output[2], engine.state.output[3],
                    engine.state.output[4]))
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, L_r:{:.3f}, L_id:{:.3f}, L_tri:{:.3f}"
                        .format(engine.state.epoch, iter,
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0],engine.state.metrics['avg_l_r'], engine.state.metrics['avg_l_id'],
                                engine.state.metrics['avg_l_tri']))

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
            # cmc, mAP = evaluator.state.metrics['r1_mAP']
            # logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            # logger.info("mAP: {:.1%}".format(mAP))
            # for r in [1, 5, 10]:
                # logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_compare_loss(engine):
        iter = engine.state.iteration
        logger.info("Evaluator----- Iteration[{}] Loss: {:.3f},  L_r:{:.3f},  L_tri:{:.3f}"
                    .format(iter,
                            engine.state.output[4], engine.state.output[5],
                            engine.state.output[6]))
        if iter % log_period == 0:
            logger.info(
                "Evaluator---- Epoch[{}] Iteration[{}] Loss: {:.3f}, Base Lr: {:.2e}, L_r:{:.3f}, L_tri:{:.3f}"
                .format(engine.state.epoch, iter,
                        engine.state.metrics['avg_loss'],
                        scheduler.get_lr()[0], engine.state.metrics['avg_l_r'],
                        engine.state.metrics['avg_l_tri']))

    # print('开始进行评价')
    # evaluator.run(val_loader)  # 只执行一个epock
    # cmc, mAP = evaluator.state.metrics['r1_mAP']
    # print(cmc,mAP)

    trainer.run(train_loader, max_epochs=epochs)

