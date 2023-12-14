import torch
import pyiqa
import argparse
import os
from PIL import Image
from collections import OrderedDict
from torchvision.transforms.functional import to_tensor
from kornia.color import rgb_to_ycbcr
from tqdm import tqdm

from utils.uciqe_uiqm import getUCIQE, getUIQM


parser = argparse.ArgumentParser()
parser.add_argument('-re_dir', '--results_dir', type=str, default='results')
parser.add_argument('--model_v', type=str, default='uie')
parser.add_argument('--net', type=str, default='erd')
parser.add_argument('--name', type=str)
parser.add_argument('--ds_name', type=str)
parser.add_argument('--epochs', type=int, nargs='+')
parser.add_argument("--load_prefix", type=str, default='weights', help="the prefix string of the filename of the weights to be loaded")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

niqe = pyiqa.create_metric('niqe', device=device)
musiq = pyiqa.create_metric('musiq', device=device)
uranker = pyiqa.create_metric('uranker', device=device)
metrics = OrderedDict(
    niqe  = {'fn': niqe, 'val': 0.0},
    musiq = {'fn': musiq, 'val': 0.0},
    uranker = {'fn': uranker, 'val': 0.0},
    uciqe = {'fn': getUCIQE, 'val':0.0},
    uiqm = {'fn': getUIQM, 'val': 0.0},
)

if not isinstance(args.epochs, list):
    args.epochs = [args.epochs]

epochs_vals = OrderedDict()

for epoch in args.epochs:
    results_dir = os.path.join(args.results_dir, args.model_v, args.net,
                                 args.name, args.ds_name, f'{args.load_prefix}_{epoch}')
    print(f'evaluating [{results_dir}]...')
    pred_imgs_dir = os.path.join(results_dir, 'single/predicted')
    noref_f = open(os.path.join(results_dir, 'noref_eval.csv'), 'w')
    noref_f.write('img_name,{}\n'.format(','.join(metrics.keys())))
    img_name_list = os.listdir(pred_imgs_dir)
    img_name_list.sort()
    for img_name in tqdm(img_name_list):
        img_path = os.path.join(pred_imgs_dir, img_name)
        img = to_tensor(Image.open(img_path)).unsqueeze(0)
        vals = []
        assert len(img.shape) == 4
        for metric_name, metric in metrics.items():
            if metric_name == 'niqe':
                val = metric['fn'](rgb_to_ycbcr(img)).item()
            elif metric_name == 'uciqe' or metric_name == 'uiqm':
                val = metric['fn'](img_path)
            else:
                val = metric['fn'](img).item()
            metric['val'] += val
            vals.append('{:.3f}'.format(val))
        noref_f.write('{},{}\n'.format(img_name, ','.join(vals)))
    avg_vals = ['{:.3f}'.format(metrics[name]['val']/len(img_name_list)) for name in metrics]
    noref_f.write('average,{}\n'.format(','.join(avg_vals)))
    noref_f.close()
    epochs_vals[epoch] = avg_vals
    for metric_name in metrics:
        metrics[metric_name]['val'] = 0.0

print('epoch\t{}'.format('\t'.join(metrics.keys())))
for epoch in epochs_vals:
    print('{}\t{}'.format(epoch, '\t'.join(epochs_vals[epoch])))