# ----------------------------------------------------------------------------------------------------------
# TS-CAM
# Copyright (c) Learning and Machine Perception Lab (LAMP), SECE, University of Chinese Academy of Science.
# ----------------------------------------------------------------------------------------------------------
import os
import sys
import datetime
import pprint
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import evaluate_cls_loc

import torch
from torch.utils.tensorboard import SummaryWriter

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
import numpy as np
from models.resnet50_cam import Net

def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=True,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
    optimizer = create_optimizer(args, model)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)

    ############################################
    # resnet cam
    ############################################
    # model2 = vgg16_cam(pretrained=True, num_classes=200)
    model2 = Net()
    param_groups2 = model2.trainable_parameters()
    optimizer2 = torch.optim.SGD([
        {'params': param_groups2[0], 'lr': 0.005, 'weight_decay': 1e-4},
        {'params': param_groups2[1], 'lr': 10*0.005, 'weight_decay': 1e-4},
    ], lr=0.005, weight_decay=1e-4)
    model2 = torch.nn.DataParallel(model2, device_ids=list(range(torch.cuda.device_count()))).to(device)
    ############################################
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)
    print('Preparing networks done!')
    return device, [model, model2], [optimizer, optimizer2], cls_criterion


def main():
    args = update_config()

    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join('ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
        cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE, cfg.BASIC.TIME))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, optimizer, cls_criterion = creat_model(cfg, args)
    best_gtknown = 0
    best_top1_loc = 0
    update_train_step = 0
    update_val_step = 0
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
        adjust_learning_rate_normal(optimizer[0], epoch, cfg)
        update_train_step, loss_train, cls_top1_train, cls_top5_train = \
            train_one_epoch(train_loader, model, device, cls_criterion,
                            optimizer, epoch, writer, cfg, update_train_step)

        update_val_step, loss_val, cls_top1_val, cls_top5_val, \
        loc_top1_val, loc_top5_val, loc_gt_known = \
            val_loc_one_epoch(val_loader, model, device, cls_criterion, epoch, writer, cfg, update_val_step)

        torch.save({
            "epoch": epoch,
            'state_dict': model[0].state_dict(),
            'best_map': best_gtknown
        }, os.path.join(ckpt_dir, 'model_epoch{}.pth'.format(epoch)))

        if loc_top1_val > best_top1_loc:
            best_top1_loc = loc_top1_val
            torch.save({
                "epoch": epoch,
                'state_dict': model[0].state_dict(),
                'best_map': best_top1_loc  # 这绝对是bug -- by Huffman
            }, os.path.join(ckpt_dir, 'model_best_top1_loc.pth'))
        if loc_gt_known > best_gtknown:
            best_gtknown = loc_gt_known
            torch.save({
                "epoch": epoch,
                'state_dict': model[0].state_dict(),
                'best_map': best_gtknown
            }, os.path.join(ckpt_dir, 'model_best.pth'))

        print("Best GT_LOC: {}".format(best_gtknown))
        print("Best TOP1_LOC: {}".format(best_top1_loc))
        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))


def train_one_epoch(train_loader, model, device, criterion, optimizer, epoch,
                    writer, cfg, update_train_step):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model[0].train()
    model[1].train()
    for i, (input, target) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)

        cls_logits, cam = model[0](input, return_cam=False)
        cls_logits1, cam1 = model[1](input, return_cam=False)
        loss1 = criterion(cls_logits, target)
        loss2 = criterion(cls_logits1, target)
        loss3 =  torch.mean(torch.abs(cam - cam1))  
        if (epoch//5) % 2 == 0:
            loss = loss1 + loss2
        else:    
            loss = loss1 + loss2 + loss3
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        loss.backward()
        optimizer[0].step()
        optimizer[1].step()

        prec1, prec5 = accuracy(cls_logits.data.contiguous(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses1.update(loss1.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        losses3.update(loss3.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        writer.add_scalar('loss_iter/train', loss.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top1', prec1.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top5', prec5.item(), update_train_step)

        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader)-1:
            print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.5f}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f}), {loss2.val:.4f} ({loss2.avg:.4f}), {loss3.val:.4f} ({loss3.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), loss=losses1, loss2=losses2, loss3=losses3,
                top1=top1, top5=top5, lr=optimizer[0].param_groups[-1]['lr'])))
    return update_train_step, losses.avg, top1.avg, top5.avg


def val_loc_one_epoch(val_loader, model, device, criterion,epoch, writer, cfg, update_val_step):

    losses = AverageMeter()

    cls_top1 = []
    cls_top5 = []
    loc_top1 = []
    loc_top5 = []
    loc_gt_known = []
    top1_loc_right = []
    top1_loc_cls = []
    top1_loc_mins = []
    top1_loc_part = []
    top1_loc_more = []
    top1_loc_wrong = []

    with torch.no_grad():
        model[0].eval()
        for i, (input, target, bbox, image_names) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input = input.to(device)
            cls_logits, cams = model[0](input, return_cam=True) 
            loss = criterion(cls_logits, target)

            prec1, prec5 = accuracy(cls_logits.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            writer.add_scalar('loss_iter/val', loss.item(), update_val_step)
            writer.add_scalar('acc_iter/val_top1', prec1.item(), update_val_step)
            writer.add_scalar('acc_iter/val_top5', prec5.item(), update_val_step)

            cls_top1_b, cls_top5_b, loc_top1_b, loc_top5_b, loc_gt_known_b, top1_loc_right_b, \
                top1_loc_cls_b,top1_loc_mins_b, top1_loc_part_b, top1_loc_more_b, top1_loc_wrong_b = \
                    evaluate_cls_loc(input, target, bbox, cls_logits, cams, image_names, cfg, epoch)
            cls_top1.extend(cls_top1_b)
            cls_top5.extend(cls_top5_b)
            loc_top1.extend(loc_top1_b)
            loc_top5.extend(loc_top5_b)
            top1_loc_right.extend(top1_loc_right_b)
            top1_loc_cls.extend(top1_loc_cls_b)
            top1_loc_mins.extend(top1_loc_mins_b)
            top1_loc_more.extend(top1_loc_more_b)
            top1_loc_part.extend(top1_loc_part_b)
            top1_loc_wrong.extend(top1_loc_wrong_b)

            loc_gt_known.extend(loc_gt_known_b)

            if i % cfg.BASIC.DISP_FREQ == 0 or i == len(val_loader)-1:
                print('Val Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i+1, len(val_loader), loss=losses))
                print('Cls@1:{0:.3f}\tCls@5:{1:.3f}\n'
                      'Loc@1:{2:.3f}\tLoc@5:{3:.3f}\tLoc_gt:{4:.3f}\n'.format(
                    list2acc(cls_top1), list2acc(cls_top5),
                    list2acc(loc_top1), list2acc(loc_top5), list2acc(loc_gt_known)))
        wrong_details = []
        wrong_details.append(np.array(top1_loc_right).sum())
        wrong_details.append(np.array(top1_loc_cls).sum())
        wrong_details.append(np.array(top1_loc_mins).sum())
        wrong_details.append(np.array(top1_loc_part).sum())
        wrong_details.append(np.array(top1_loc_more).sum())
        wrong_details.append(np.array(top1_loc_wrong).sum())
        print('wrong_details:{} {} {} {} {} {}'.format(wrong_details[0], wrong_details[1], wrong_details[2],
                                                       wrong_details[3], wrong_details[4], wrong_details[5]))
    return update_val_step, losses.avg, list2acc(cls_top1), list2acc(cls_top5), \
           list2acc(loc_top1), list2acc(loc_top5), list2acc(loc_gt_known)
if __name__ == "__main__":
    main()
