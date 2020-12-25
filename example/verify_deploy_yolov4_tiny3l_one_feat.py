# encoding=utf-8
import sys

sys.path.insert(0, '.')
sys.path.append('/users/maqiao/DNN/caffes/caffe_mq/python')
# sys.path.append('/mnt/diskb/even/MCMOT/src')
# from lib.models.model import create_model, load_model
sys.path.append('/mnt/diskb/even/YOLOV4')
sys.path.append('.')

import re
from functools import cmp_to_key
from models import *
from utils.datasets import *
from utils.utils import *

import caffe
import torchvision.transforms as transforms
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from model import resnet
import time

import cv2


# caffe load formate
def load_image_caffe(img_file):
    image = caffe.io.load_image(img_file)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image


def load_image_pytorch(img_file):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # # transforms.ToTensor()
    # transform1 = transforms.Compose([
    #     transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    #     normalize
    #     ]
    # )
    # ##numpy.ndarray
    # img = cv2.imread(imgfile)# 读取图像
    # img = cv2.resize(img,(224,244))
    # img1 = transform1(img) # 归一化到 [0.0,1.0]
    # print("img1 = ",img1)

    img = np.ones([1, 3, args.height, args.width])

    # 转化为numpy.ndarray并显示
    return img


def forward_pytorch(cfg_file, weight_file, image):
    # define model network
    # net = resnet.resnet18()
    # checkpoint = torch.load(weight_file)
    # net.load_state_dict(checkpoint)

    # heads = {'hm': 5,
    #          'wh': 2,
    #          'reg': 2,
    #          'id': 128}
    # net = create_model(arch='resdcn_18', heads=heads, head_conv=-1)
    # # model_path = '/mnt/diskc/maqiao/even/MCMOT/exp/mot/default/mcmot_last_track_resdcn_18.pth'
    # net = load_model(model=net, model_path=weight_file)
    # net.eval()

    names = ['car', 'bicycle', 'person', 'cyclist', 'tricycle']
    opt = {
        'device': 'cpu',  # '0'
        'img_size': 768,
        'feat_out_ids': '-1',
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
    net = Darknet(cfg_file,
                  opt['img_size'],
                  False,
                  max_id_dict,
                  128,
                  'FC',
                  opt['feat_out_ids'],
                  'track').to(device)
    print('{} cfg file loaded.'.format(cfg_file))

    # load weight file(.pt or .weights)
    if not os.path.isfile(weight_file):
        print('[Err]: invalid weight file.')
        exit(-1)
    if weight_file.endswith('.pt'):  # pytorch format
        ckpt = torch.load(weight_file, map_location=device)
        net.load_state_dict(ckpt['model'])
        if 'epoch' in ckpt.keys():
            print('Checkpoint of epoch {} loaded.'.format(ckpt['epoch']))
    elif len(weight_file) > 0:  # dark-net format
        load_darknet_weights(net, weight_file, cutoff=0)
        print('{} loaded.'.format(weight_file))

    if args.cuda:
        net.cuda()

    net.eval()
    print(net)

    image = torch.from_numpy(image)
    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)

    image = torch.tensor(image, dtype=torch.float32)

    t0 = time.time()

    blobs = net.forward(image)
    # print(blobs.data.numpy().flatten())

    t1 = time.time()
    return t1 - t0, blobs, net.parameters()


# Reference from:
def forward_caffe(proto_file, weight_file, image):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(proto_file, weight_file, caffe.TEST)
    net.blobs['blob1'].reshape(1, 3, args.height, args.width)
    net.blobs['blob1'].data[...] = image

    t0 = time.time()

    output = net.forward()

    t1 = time.time()

    return t1 - t0, net.blobs, net.params
    # return t1 - t0, output, net.params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')

    # ---------- Caffe cfg and weight file
    parser.add_argument('--caffecfg', default='mcmot_yolov4_tiny3l_one_feat_fuse.prototxt', type=str)
    parser.add_argument('--caffeweight', default='mcmot_yolov4_tiny3l_one_feat_fuse.caffemodel', type=str)

    # ---------- Pytorch cfg and weight file
    parser.add_argument('--pytorchcfg',
                        type=str,
                        default='/mnt/diskb/even/YOLOV4/cfg/yolov4-tiny-3l_no_group_id_one_feat_fuse.cfg')
    parser.add_argument('--pytorchweight',
                        default='/mnt/diskb/even/YOLOV4/weights/v4_tiny3l_one_feat_fuse_track_last.weights',
                        type=str)

    parser.add_argument('--imgfile', default='001763.jpg', type=str)
    parser.add_argument('--height', default=448, type=int)
    parser.add_argument('--width', default=768, type=int)
    parser.add_argument('--meanB', default=104, type=float)
    parser.add_argument('--meanG', default=117, type=float)
    parser.add_argument('--meanR', default=123, type=float)
    parser.add_argument('--scale', default=255, type=float)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    print(args)

    # Caffe cfg and weight file
    caffe_cfg_file = args.caffecfg
    caffe_weight_file = args.caffeweight

    # Pytorch cfg and weight file
    pytorch_cfg_file = args.pytorchcfg
    pytorch_weight_file = args.pytorchweight

    img_file = args.imgfile

    # load image data
    image = load_image_pytorch(img_file)

    time_pytorch, out_Tensor_pytorch, pytorch_models = forward_pytorch(pytorch_cfg_file, pytorch_weight_file, image)
    time_caffe, out_Tensor_caffe, caffe_params = forward_caffe(caffe_cfg_file, caffe_weight_file, image)

    print('pytorch forward time {:.3f}'.format(time_pytorch))
    print('caffe forward time {:.3f}'.format(time_caffe))

    print('------------ Output Difference ------------')
    # blob_name = 'fc_blob1'

    # No-upsample: compare 6 layers: 3 yolo output layers and 3 feature layers
    layer_names = ['conv_blob18', 'conv_blob21', 'conv_blob24',
                   'batch_norm_blob23']
    caffe_layer_0 = out_Tensor_caffe[layer_names[0]].data
    caffe_layer_1 = out_Tensor_caffe[layer_names[1]].data
    caffe_layer_2 = out_Tensor_caffe[layer_names[2]].data
    caffe_layer_3 = out_Tensor_caffe[layer_names[3]].data

    if args.cuda:
        pytorch_layer_0 = out_Tensor_pytorch[0].data.cpu().numpy()
        pytorch_layer_1 = out_Tensor_pytorch[1].data.cpu().numpy()
        pytorch_layer_2 = out_Tensor_pytorch[2].data.cpu().numpy()
        pytorch_layer_3 = out_Tensor_pytorch[3].data.cpu().numpy()

    else:
        pytorch_layer_0 = out_Tensor_pytorch[0].data.numpy()
        pytorch_layer_1 = out_Tensor_pytorch[1].data.numpy()
        pytorch_layer_2 = out_Tensor_pytorch[2].data.numpy()
        pytorch_layer_3 = out_Tensor_pytorch[3].data.numpy()

    layer_diff_0 = abs(pytorch_layer_0 - caffe_layer_0).sum() / pytorch_layer_0.size  # numpy size
    layer_diff_1 = abs(pytorch_layer_1 - caffe_layer_1).sum() / pytorch_layer_1.size
    layer_diff_2 = abs(pytorch_layer_2 - caffe_layer_2).sum() / pytorch_layer_2.size
    layer_diff_3 = abs(pytorch_layer_3 - caffe_layer_3).sum() / pytorch_layer_3.size

    print('{:s} diff: {:.3f}'.format(layer_names[0], layer_diff_0))
    print('{:s} diff: {:.3f}'.format(layer_names[1], layer_diff_1))
    print('{:s} diff: {:.3f}'.format(layer_names[2], layer_diff_2))
    print('{:s} diff: {:.3f}'.format(layer_names[3], layer_diff_3))

    print('Done.')
