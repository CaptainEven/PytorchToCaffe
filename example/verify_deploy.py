# encoding=utf-8
import sys

sys.path.insert(0, '.')
sys.path.append('/users/maqiao/DNN/caffes/caffe_mq/python')
sys.path.append('/mnt/diskb/even/MCMOT/src')
from lib.models.model import create_model, load_model

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


def forward_pytorch(weight_file, image):
    # define model network
    # net = resnet.resnet18()
    checkpoint = torch.load(weight_file)
    # net.load_state_dict(checkpoint)

    heads = {'hm': 5,
             'wh': 2,
             'reg': 2,
             'id': 128}
    net = create_model(arch='resdcn_18', heads=heads, head_conv=-1)
    # model_path = '/mnt/diskc/maqiao/even/MCMOT/exp/mot/default/mcmot_last_track_resdcn_18.pth'
    net = load_model(model=net, model_path=weight_file)
    net.eval()

    if args.cuda:
        net.cuda()

    print(net)
    net.eval()
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
    # return t1 - t0, net.blobs, net.params
    return t1 - t0, output, net.params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--protofile', default='mcmot_resdcn_18.prototxt', type=str)
    parser.add_argument('--weightfile', default='mcmot_resdcn_18.caffemodel', type=str)
    parser.add_argument('--model',
                        default='/mnt/diskb/even/MCMOT/exp/mot/default/mcmot_last_track_resdcn_18.pth',
                        type=str)
    parser.add_argument('--imgfile', default='001763.jpg', type=str)
    parser.add_argument('--height', default=608, type=int)
    parser.add_argument('--width', default=1088, type=int)
    parser.add_argument('--meanB', default=104, type=float)
    parser.add_argument('--meanG', default=117, type=float)
    parser.add_argument('--meanR', default=123, type=float)
    parser.add_argument('--scale', default=255, type=float)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    print(args)

    protofile = args.protofile
    weightfile = args.weightfile
    imgfile = args.imgfile

    image = load_image_pytorch(imgfile)
    time_pytorch, out_Tensor_pytorch, pytorch_models = forward_pytorch(args.model, image)
    time_caffe, out_Tensor_caffe, caffe_params = forward_caffe(protofile, weightfile, image)

    print('pytorch forward time %d', time_pytorch)
    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    blob_name = 'fc_blob1'

    # print(out_Tensor_caffe)
    caffe_data = out_Tensor_caffe[blob_name].data[0][...].flatten()

    if args.cuda:
        pytorch_data = out_Tensor_pytorch.data.cpu().numpy().flatten()
    else:
        pytorch_data = out_Tensor_pytorch.data.numpy().flatten()

    diff = abs(pytorch_data - caffe_data).sum()
    print('%-30s pytorch_shape: %-20s caffe_shape: %-20s output_diff: %f' % (
        blob_name, pytorch_data.shape, caffe_data.shape, diff / pytorch_data.size))
