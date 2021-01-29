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
        'device': 'cpu',  # '0'
        'img_size': 768,
        'feat_out_ids': '-1',
        'cfg': '/mnt/diskb/even/YOLOV4/cfg/yolov4_half_one_feat_fuse.cfg',
        'weights': '/mnt/diskb/even/YOLOV4/weights/one_feat_fuse_track_last.weights',
    }
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    ## read from .npy(max_id_dict.npy file)
    max_id_dict_file_path = '/mnt/diskb/even/dataset/MCMOT/max_id_dict.npz'
    if os.path.isfile(max_id_dict_file_path):
        load_dict = np.load(max_id_dict_file_path, allow_pickle=True)
    max_id_dict = load_dict['max_id_dict'][()]
    print(max_id_dict)

    device = torch_utils.select_device(opt['device'])
    net = Darknet(opt['cfg'],
                  opt['img_size'],
                  False,
                  max_id_dict,
                  128,
                  'FC',
                  opt['feat_out_ids'],
                  'track').to(device)

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
        load_darknet_weights(net, opt['weights'], cutoff=0)

    net.to(device).eval()

    input = torch.ones([1, 3, 448, 768])
    name = 'mcmot_yolo_half_one_feat_fuse'

    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
