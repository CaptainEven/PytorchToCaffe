# encoding=utf-8
import os
import sys

sys.path.append('/mnt/diskb/even/YOLOV4')
sys.path.append('.')

from models import *
from utils.datasets import *
from utils.utils import *
# from tracker.multitracker import JDETracker
# from tracking_utils import visualization as vis

import pytorch_to_caffe
import torch
from collections import defaultdict

if __name__ == '__main__':
    names = ['car', 'bicycle', 'person', 'cyclist', 'tricycle']
    opt = {
        'img_size': 768,
        'cfg': '/mnt/diskb/even/YOLOV4/cfg/yolov4_half-mcmot.cfg',
        'device': 'cpu',  # '0'
        'weights': '/mnt/diskb/even/YOLOV4/weights/pure_detect_last.pt',
    }
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    max_ids_dict = {
        0: 330,           # car
        1: 102,           # bicycle
        2: 104,           # person
        3: 312,           # cyclist
        4: 53             # tricycle
    }  # cls_id -> track id number for traning

    device = torch_utils.select_device(opt['device'])
    net = Darknet(opt['cfg'], opt['img_size'], False, max_ids_dict, 128, 'track').to(device)

    # load weight file(.pt or .weights)
    if not os.path.isfile(opt['weights']):
        print('[Err]: invalid weight file.')
        exit(-1)
    if opt['weights'].endswith('.pt'):  # pytorch format
        ckpt = torch.load(opt['weights'], map_location=device)
        net.load_state_dict(ckpt['model'])
        if 'epoch' in ckpt.keys():
            print('Checkpoint of epoch {} loaded.'.format(ckpt['epoch']))
    elif len(opt['weights']) > 0:  # darknet format
        load_darknet_weights(net, opt['weights'])

    net.to(device).eval()

    input = torch.ones([1, 3, 448, 768])
    name = 'mcmot_yolov4_half'

    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
