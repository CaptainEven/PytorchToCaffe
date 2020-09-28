# encoding=utf-8
import sys

sys.path.append('/mnt/diskb/even/MCMOT/src')
sys.path.append('.')
import pytorch_to_caffe
# from torch.autograd import Variable
import torch
from lib.models.model import create_model, load_model


if __name__ == '__main__':
    heads = {'hm': 5,
             'wh': 2,
             'reg': 2,
             'id': 128}
    net = create_model(arch='hrnet_18', heads=heads, head_conv=-1)
    model_path = '/mnt/diskc/maqiao/even/MCMOT/exp/mot/default/mcmot_last_det_hrnet_18_de_conv.pth'
    net = load_model(model=net, model_path=model_path)
    net.eval()

    input = torch.ones([1, 3, 608, 1088])
    name = 'mcmot_hrnet18'

    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
