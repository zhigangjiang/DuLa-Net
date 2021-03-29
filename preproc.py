import os
import argparse
import numpy as np
from pano import pano_connect_points
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir',  help='dataset root dir', default='/Users/jiangzhigang/Code/pytorch-layoutnet/data')
parser.add_argument('--datasets', help='dataset names', default=['train', 'test', 'valid'])
parser.add_argument('--H', default=512)
parser.add_argument('--W', default=1024)
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)


def get_mfc(ceil_coor, floor_coor):

    y0 = np.zeros(args.W)
    y1 = np.zeros(args.W)

    for j in range(ceil_coor.shape[0]):
        coorxy = pano_connect_points(ceil_coor[j], ceil_coor[(j + 1) % 4], -50)
        y0[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(floor_coor[j], floor_coor[(j + 1) % 4], 50)
        y1[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

    surface = np.zeros((args.H, args.W), dtype=np.int32)
    surface[np.round(y0).astype(int), np.arange(args.W)] = 1
    surface[np.round(y1).astype(int), np.arange(args.W)] = -1
    surface = np.cumsum(surface, axis=0) * 255

    return surface


def pre_process():
    for dataset in args.datasets:
        dataset_path = os.path.join(args.root_dir, dataset)
        img_dir = os.path.join(dataset_path, 'img')
        mfc_dir = os.path.join(dataset_path, 'mfc')
        label_dir = os.path.join(dataset_path, 'label_cor')
        if not os.path.isdir(mfc_dir):
            os.makedirs(mfc_dir)

        for img_name in os.listdir(img_dir):
            if img_name.startswith('.'):
                continue
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, img_name.replace('.png', '.txt'))
            mfc_path = os.path.join(mfc_dir, img_name)
            with open(label_path) as f:
                gt = np.array([line.strip().split() for line in f], np.float64)
            mfc = get_mfc(gt[0::2], gt[1::2])
            cv2.imwrite(mfc_path, mfc)

if __name__ == '__main__':
    pre_process()