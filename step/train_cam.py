
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            cam = model(img)
            loss1 = F.multilabel_soft_margin_loss(torchutils.gap2d(cam), label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


def run(args):

    model1 = getattr(importlib.import_module(args.cam_network_branch1), 'Net')()
    model2 = getattr(importlib.import_module(args.cam_network_branch2), 'Net')(pretrained=True, num_classes=20)

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(210, 420), hor_flip=True,
                                                                crop_size=336, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=336)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups1 = model1.trainable_parameters()
    optimizer1 = torchutils.PolyOptimizer([
        {'params': param_groups1[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups1[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    param_groups2 = model2.trainable_parameters()
    optimizer2 = torchutils.PolyOptimizer([
        {'params': param_groups2[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups2[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model1 = torch.nn.DataParallel(model1).cuda()
    model1.train()
    model2 = torch.nn.DataParallel(model2).cuda()
    model2.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            cam1 = model1(img)
            cam2 = model2(img)

            loss1 = F.multilabel_soft_margin_loss(torchutils.gap2d(cam1), label)
            loss2 = F.multilabel_soft_margin_loss(torchutils.gap2d(cam2), label)
            cam1 = F.interpolate(cam1, size=(336, 336), mode='bilinear', align_corners=True)
            cam2 = F.interpolate(cam2, size=(336, 336), mode='bilinear', align_corners=True)
            loss3 = torch.mean(torch.abs(cam1 - cam2))

            if ep in [0]:
                loss = loss1 + loss2
            else:
                loss = loss1 + loss2 + args.cam_lambda * loss3

            avg_meter.add({'loss1': loss1.item()})
            avg_meter.add({'loss2': loss2.item()})
            avg_meter.add({'loss3': loss3.item()})

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            if (optimizer1.global_step-1)%100 == 0:
                timer.update_progress(optimizer1.global_step / max_step)

                print('step:%5d/%5d' % (optimizer1.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f' % (avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer1.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model1, val_data_loader)
            timer.reset_stage()

    torch.save(model1.module.state_dict(), args.cam1_weights_name + '.pth')
    torch.save(model2.module.state_dict(), args.cam2_weights_name + '.pth')
    torch.cuda.empty_cache()