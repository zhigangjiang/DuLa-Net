import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import PanoDataset
import os
import argparse
import numpy as np
import torch
import config as cf
from Model import DuLaNet, E2P
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import Utils.tools as tools
import shutil

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--id', required=True,
                    help='experiment id to name checkpoints')
parser.add_argument('--mode', required=True, choices=['train', 'continue'],
                    help='the run mode')
parser.add_argument('--backbone', default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50'], help='backbone network')
parser.add_argument('--ckpt', default='./Model/ckpt',
                    help='folder to output checkpoints')
parser.add_argument('--ckpt_path', default='',
                    help='checkpoint path for continue train')
# Dataset related arguments
parser.add_argument('--root_dir_train', default='data/train',
                    help='root directory for training data')
parser.add_argument('--root_dir_valid', default='data/valid',
                    help='root directory for validation data')
parser.add_argument('--input_cat', default=['img'], nargs='+',
                    help='input channels subdirectories')
parser.add_argument('--input_channels', default=3, type=int,
                    help='numbers of input channels')
parser.add_argument('--no_flip', action='store_true',
                    help='disable left-right flip augmentation')
parser.add_argument('--no_rotate', action='store_true',
                    help='disable horizontal rotate augmentation')
parser.add_argument('--no_gamma', action='store_true',
                    help='disable gamma augmentation')
parser.add_argument('--noise', action='store_true',
                    help='enable noise augmentation')
parser.add_argument('--contrast', action='store_true',
                    help='enable contrast augmentation')
parser.add_argument('--num_workers', default=0, type=int,
                    help='numbers of workers for dataloaders')
# optimization related arguments
parser.add_argument('--batch_size_train', default=12, type=int,
                    help='training mini-batch size')
parser.add_argument('--batch_size_valid', default=12, type=int,
                    help='validation mini-batch size')
parser.add_argument('--epochs', default=100, type=int,
                    help='epochs to train')
parser.add_argument('--optim', default='Adam',
                    help='optimizer to use. only support SGD and Adam')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--lr_pow', default=0, type=float,
                    help='power in poly to drop LR')
parser.add_argument('--warmup_lr', default=1e-6, type=float,
                    help='starting learning rate for warm up')
parser.add_argument('--warmup_epochs', default=10, type=int,
                    help='numbers of warmup epochs')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='factor for L2 regularization')
parser.add_argument('--cormap_smooth', default=0, type=float,
                    help='cor probability smooth constraint')
# Misc arguments
parser.add_argument('--no_cuda', action='store_true',
                    help='disable cuda')
parser.add_argument('--seed', default=277, type=int,
                    help='manual seed')
parser.add_argument('--disp_iter', type=int, default=500,
                    help='iterations frequency to display')
parser.add_argument('--save_every', type=int, default=5,
                    help='epochs frequency to save state_dict')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda
                      else 'cpu')
print('device:{}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)

# Create dataloader
dataset_train = PanoDataset(root_dir=args.root_dir_train,
                            cat_list=[*args.input_cat, 'mfc'],
                            flip=not args.no_flip, rotate=not args.no_rotate,
                            gamma=not args.no_gamma, noise=args.noise,
                            contrast=args.contrast)
dataset_valid = PanoDataset(root_dir=args.root_dir_valid,
                            cat_list=[*args.input_cat, 'mfc'],
                            flip=False, rotate=False,
                            gamma=False, noise=False,
                            contrast=False)
loader_train = DataLoader(dataset_train, args.batch_size_train,
                          shuffle=True, drop_last=True,
                          num_workers=args.num_workers,
                          pin_memory=not args.no_cuda)
loader_valid = DataLoader(dataset_valid, args.batch_size_valid,
                          shuffle=False, drop_last=False,
                          num_workers=args.num_workers,
                          pin_memory=not args.no_cuda)
log_dir = "./run/{}".format(args.id)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)
writer_dict = {
    "train_loss": [],
    "valid_loss": [],
    "running_lr": []
}

# Create model
model = DuLaNet(args.backbone, gpu=True).to(device)

# Create optimizer
if args.optim == 'SGD':
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
else:
    raise NotImplementedError()

start_epoch = 0
min_loss = 1e10

if args.mode == "continue" and len(args.ckpt_path) != 0:
    print("load checkpoint path:{}".format(args.ckpt_path))
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']
    min_loss = checkpoint['min_loss']
    writer_dict = checkpoint['writer_dict']
    for t in writer_dict.keys():
        for i, s in enumerate(writer_dict[t]):
            writer.add_scalar(t, s, global_step=i)

# if args.mode == "continue" and len(args.ckpt_path) != 0:
#     checkpoint = torch.load(args.ckpt_path)
#     model.load_state_dict(checkpoint)


# Init variable
args.warmup_iters = args.warmup_epochs * len(loader_train)
args.max_iters = args.epochs * len(loader_train)
args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
args.cur_iter = start_epoch * len(loader_train)

BCELoss = nn.BCELoss(reduction='mean')
L1Loss = nn.L1Loss(reduction='mean')

print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

print('%d iters per epoch for train' % len(loader_train))
print('%d iters per epoch for valid' % len(loader_valid))
print(' start training '.center(80, '='))

e2p = E2P(cf.pano_size, cf.fp_size, cf.fp_fov, gpu=True)

# Start training
for ith_epoch in range(start_epoch, args.epochs + 1):
    model.train()
    torch.set_grad_enabled(True)
    train_loss = 0.0
    for ith_batch, datas in enumerate(loader_train):
        # Set learning rate
        tools.adjust_learning_rate(optimizer, args)
        writer.add_scalar('running_lr', args.running_lr, global_step=args.cur_iter)
        writer_dict['running_lr'].append(args.running_lr)

        args.cur_iter += 1

        # Prepare data
        x = torch.cat([datas[i]
                       for i in range(len(args.input_cat))], dim=1).to(device)
        h = datas[-1].to(device)
        fc = datas[-2].to(device)

        [fp, fp_down] = e2p(fc)

        # Feedforward
        [fp_, fc_, h_] = model(x)

        # Compute loss
        loss_fc = BCELoss(fc_, fc)
        loss_fp = BCELoss(fp_, fp)
        loss_h = L1Loss(h_, h)
        loss = loss_fc + loss_fp + 0.5 * loss_h

        # backprop
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        # nn.utils.clip_grad_norm_(chain(
        #     encoder.parameters(), edg_decoder.parameters(), cor_decoder.parameters()),
        #     3.0, norm_type='inf')
        optimizer.step()

        # Statical result
        train_loss += loss
        print('\rcur_iter %d | ith_batch %d (epoch %d) | lr %.6f | loss %s' % (
            args.cur_iter, ith_batch, ith_epoch, args.running_lr, loss), end='', flush=True)

        if args.cur_iter % args.disp_iter == 0:
            ff = fc_.cpu()
            fs = fp_.cpu()
            plt.imshow(ff[0][0].detach().numpy())
            plt.show()
            plt.imshow(fs[0][0].detach().numpy())
            plt.show()

    # Validate
    model.eval()
    torch.set_grad_enabled(False)
    valid_loss = 0.0
    for ith_batch, datas in enumerate(loader_valid):
        with torch.no_grad():
            # Prepare data
            x = torch.cat([datas[i]
                           for i in range(len(args.input_cat))], dim=1).to(device)
            h = datas[-1].to(device)
            fc = datas[-2].to(device)
            [fp, fp_down] = e2p(fc)

            # Feedforward
            [fp_, fc_, h_] = model(x)

            # Compute loss
            loss_fc = BCELoss(fc_, fc)
            loss_fp = BCELoss(fp_, fp)
            loss_h = L1Loss(h_, h)
            loss = loss_fc + loss_fp + 0.5 * loss_h

            valid_loss += loss.item()

    train_loss /= len(loader_train)
    valid_loss /= len(loader_valid)

    print("\n")
    print('validation | epoch %d | train %s | valid %s' % (ith_epoch, train_loss, valid_loss))
    writer.add_scalar('train_loss', train_loss, global_step=ith_epoch)
    writer_dict['train_loss'].append(train_loss)
    writer.add_scalar('valid_loss', valid_loss, global_step=ith_epoch)
    writer_dict['valid_loss'].append(valid_loss)

    # Dump model
    if valid_loss < min_loss:
        min_loss = valid_loss
        checkpoint = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": ith_epoch,
            'min_loss': min_loss,
            'writer_dict': writer_dict
        }
        model_path = os.path.join(args.ckpt, args.id, '%s_epoch_%d_%.2f.pth' % (args.backbone, ith_epoch, min_loss))
        torch.save(checkpoint, model_path)
        print('model data saved: %s' % model_path)



