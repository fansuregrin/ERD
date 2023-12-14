import argparse
import yaml
import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from models import create_model
from data import (
    create_train_dataset, create_val_dataset,
    create_train_dataloader, create_val_dataloader
)
from utils import (
    seed_everything,
    LOGURU_FORMAT
)


# Command-line options and arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ds_cfg", type=str, default="configs/dataset/lsui.yaml")
parser.add_argument("--lr_scheduler_cfg", type=str, default="configs/lr_scheduler/step_lr_03.yaml")
parser.add_argument("--net_cfg", type=str, default="configs/network/erd_15blocks_2down.yaml")
parser.add_argument("--name", type=str, default="experiment", help="name of training process")
parser.add_argument("--start_epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--start_iteration", type=int, default=0, help="which iteration to start from")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs for training")
parser.add_argument("--batch_size", type=int, default=8, help="size of batches")
parser.add_argument("--seed", type=int, default=2023, help="lucky random seed")
parser.add_argument("--optimizer", type=str, default='adam', help='which optimizer to use')
parser.add_argument("--lr", type=float, default=0.001, help="learning rate for optimizer")
parser.add_argument("--lambda_mae", type=float, default=1.0, help="weight of MAE loss")
parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight of SSIM loss")
parser.add_argument("--lambda_four", type=float, default=0.5, help="weight of FourDomain loss")
parser.add_argument("--lambda_edge", type=float, default=0.5, help="weight of Edge loss")
parser.add_argument("--val_interval", type=int, default=100, help="how many iterations to validate the model")
parser.add_argument("--ckpt_interval", type=int, default=5, help="how many epochs to save checkpoint")
parser.add_argument("--l1_reduction", type=str, default='mean')
args = parser.parse_args()

model_v = 'uie'

# Dataset config
with open(args.ds_cfg) as f:
    ds_cfg = yaml.load(f, yaml.FullLoader)
# Learning rate scheduler config
with open(args.lr_scheduler_cfg) as f:
    lr_scheduler_cfg = yaml.load(f, yaml.FullLoader)
# Network config
with open(args.net_cfg) as f:
    net_cfg = yaml.load(f, yaml.FullLoader)

net_name = net_cfg['name']

# Create some useful directories
samples_dir = "samples/{}/{}/{}".format(model_v, net_name, args.name)
checkpoint_dir = "checkpoints/{}/{}/{}/".format(model_v, net_name, args.name)
log_dir = os.path.join(checkpoint_dir, 'logs')
tensorboard_log_dir = os.path.join('runs', model_v, net_name, args.name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize logger
logger.remove(0)
logger.add(sys.stdout, format=LOGURU_FORMAT)
logger.add(os.path.join(log_dir, "train_{time}.log"), format=LOGURU_FORMAT)

# Write some training infomation into log file
logger.info(f"Starting Training Process...")
logger.info(f"net_name: {net_name}")
logger.info(f"samples_dir: {samples_dir}")
logger.info(f"checkpoint_dir: {checkpoint_dir}")
logger.info(f"log_dir: {log_dir}")
logger.info(f"tensorboard_log_dir: {tensorboard_log_dir}")
for option, value in vars(args).items():
    logger.info(f"{option}: {value}")
for option, value in ds_cfg.items():
    logger.info(f"{option}: {value}")
for option, value in lr_scheduler_cfg.items():
    logger.info(f"{option}: {value}")

# Set random seed
seed_everything(args.seed)

# Set device for pytorch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


# Data pipeline
train_ds = create_train_dataset('paired_img', ds_cfg['train'])
train_dl_cfg = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 4,
}
train_dl = create_train_dataloader(train_ds, train_dl_cfg)
val_ds = create_val_dataset('paired_img', ds_cfg['val'])
val_dl_cfg = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 4,
}
val_dl = create_val_dataloader(val_ds, val_dl_cfg)

# Create a tensorboard writer
tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)

# Create and initialize model
model_cfg = {
    'mode': 'train',
    'device': DEVICE,
    'logger': logger,
    'tb_writer': tensorboard_writer,
    'sample_dir': samples_dir,
    'checkpoint_dir': checkpoint_dir,
    'name': args.name,
    'start_epoch': args.start_epoch,
    'start_iteration': args.start_iteration,
    'num_epochs': args.num_epochs,
    'val_interval': args.val_interval,
    'ckpt_interval': args.ckpt_interval,
    'optimizer': {
        'name': args.optimizer,
        'lr': args.lr,
    },
    'lr_scheduler': lr_scheduler_cfg,
    'lambda_mae': args.lambda_mae,
    'lambda_ssim': args.lambda_ssim,
    'lambda_four': args.lambda_four,
    'lambda_edge': args.lambda_edge,
    'net_cfg': net_cfg,
    'l1_reduction': args.l1_reduction,
}
model = create_model(model_v, model_cfg)

model.train(train_dl, val_dl)