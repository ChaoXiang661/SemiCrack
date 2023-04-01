import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator, RandomGenerator1,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, hausdorff, contrastive_loss
from val_2D import test_single_volume
from networks.csdnet import classifier, projectors

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/c500_256', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='c500_256/Interpolation_Training', help='experiment_name')
parser.add_argument('--model1', type=str,
                    default='transunet', help='model_name')
parser.add_argument('--model2', type=str,
                    default='transunet', help='model_name')
parser.add_argument('--max_epoch', type=int,
                    default=500, help='maximum number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=157, help='random seed')
parser.add_argument('--num_classes', type=int,  default=1,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=55,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=400.0, help='consistency_rampup')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_epoch * (args.labeled_num / args.labeled_bs)

    def create_model1(ema=False):
        # Network definition
        model = net_factory(net_type=args.model1, in_chns=3, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    def create_model2(ema=False):
        # Network definition
        model = net_factory(net_type=args.model2, in_chns=3, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model1 = create_model1()
    model2 = create_model2()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val", transform=transforms.Compose([
        RandomGenerator1(args.patch_size)
    ]))

    total_slices = len(db_train)
    labeled_slice = args.labeled_num
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    projector_1 = projectors().cuda()
    projector_2 = projectors().cuda()
    classifier_1 = classifier().cuda()
    classifier_2 = classifier().cuda()

    model1.train()
    model2.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)

    ce_loss = nn.BCELoss()
    dice_loss = losses.DiceLoss()
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss_bd = hausdorff.HausdorffDTLoss()
    c1_loss = contrastive_loss.ConLoss()
    c2_loss = contrastive_loss.contrastive_loss_sup()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = args.max_epoch
    best_performance1 = 0.0
    best_performance2 = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            label_batch = label_batch[0:args.labeled_bs, ...]
            noise1 = torch.clamp(torch.randn_like(labeled_volume_batch) * 0.04, -0.04, 0.04)
            labeled_volume_batch1 = labeled_volume_batch + noise1
            noise2 = torch.clamp(torch.randn_like(labeled_volume_batch) * 0.08, -0.08, 0.08)
            labeled_volume_batch2 = labeled_volume_batch + noise2

            # Train Model 1
            prediction_1, prediction_1_1 = model1(labeled_volume_batch1)

            feat_1, u_prediction_1 = model1(unlabeled_volume_batch)
            # print(labeled_volume_batch)

            # Train Model 2
            prediction_2, prediction_2_2 = model2(labeled_volume_batch2)

            feat_2, u_prediction_2 = model2(unlabeled_volume_batch)
            # print(prediction_2.shape)

            feat_q = projector_1(feat_1)
            feat_k = projector_2(feat_2)
            feat_l_q = classifier_1(prediction_1[0:args.labeled_bs//2])
            feat_l_k = classifier_2(prediction_2[args.labeled_bs//2:])

            loss_dice1 = dice_loss(prediction_1_1, label_batch.unsqueeze(1))
            loss_dice2 = dice_loss(prediction_2_2, label_batch.unsqueeze(1))
            supervised_loss = loss_dice1 + loss_dice2
            consistency_weight = get_current_consistency_weight(epoch_num)
            consistency_loss1 = 0.5 * (dice_loss(u_prediction_2, u_prediction_1) + dice_loss(u_prediction_1, u_prediction_2))

            consistency_loss2_1 = c1_loss(feat_q, feat_k)
            consistency_loss2_2 = c2_loss(feat_l_q, feat_l_k)
            consistency_loss2 = consistency_loss2_1 + consistency_loss2_2
            loss = supervised_loss + consistency_loss1 + consistency_weight * consistency_loss2

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 3 #0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
            writer.add_scalar('info/consistency_loss1',
                              consistency_loss1, iter_num)
            writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)
            writer.add_scalar('info/consistency_loss2',
                              consistency_loss2, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %.5f, loss_dice1: %.5f, consistency_loss1: %.5f, loss_dice2: %.5f, consistency_loss2_1: %.5f, consistency_loss2_2: %.5f' %
                (iter_num, loss.item(),  loss_dice1.item(), consistency_loss1.item(),  loss_dice2.item(), consistency_loss2_1.item(), consistency_loss2_2.item()))   # loss_ce: %f,loss_ce.item(),

        if epoch_num > 0:
            model1.eval()
            model2.eval()
            metric_list1 = 0.0
            metric_list2 = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch = volume_batch.cuda()
                metric_i1 = test_single_volume(volume_batch, label_batch, model1, args.model1)
                metric_i2 = test_single_volume(volume_batch, label_batch, model2, args.model2)
                metric_list1 += np.array(metric_i1)
                metric_list2 += np.array(metric_i2)

            metric_list1 = metric_list1 / len(db_val)
            metric_list2 = metric_list2 / len(db_val)

            performance1 = np.mean(metric_list1)
            performance2 = np.mean(metric_list2)

            if performance1 > best_performance1:
                best_performance1 = performance1
                save_mode_path = os.path.join(snapshot_path,
                                              '1epoch_{}_dice_{}.pth'.format(
                                                  epoch_num, round(best_performance1, 4)))
                save_best = os.path.join(snapshot_path,
                                         '{}_best_model1.pth'.format(args.model1))
                torch.save(model1.state_dict(), save_mode_path)
                torch.save(model1.state_dict(), save_best)
            if performance2 > best_performance2:
                best_performance2 = performance2
                save_mode_path = os.path.join(snapshot_path,
                                              '2epoch_{}_dice_{}.pth'.format(
                                                  epoch_num, round(best_performance2, 4)))
                save_best = os.path.join(snapshot_path,
                                         '{}_best_model2.pth'.format(args.model2))
                torch.save(model2.state_dict(), save_mode_path)
                torch.save(model2.state_dict(), save_best)

            logging.info(
                'epoch %d : mean_dice1 : %f  mean_dice2 : %f ' % (epoch_num, performance1, performance2))  # mean_hd95 : %f, mean_hd95
            model1.train()
            model2.train()

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model1)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
