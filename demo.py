import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import matplotlib.pyplot as plt
img = read_image('misc/demo.jpg')
img = t.from_numpy(img)[None]
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('/workspace/data/simple-faster-rcnn-pytorch/model/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth.701052458187')
opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
print("_bboxes",_bboxes)

vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))
# print("result.shape",result.shape)
plt.savefig('result.jpg')
# it failed to find the dog, but if you set threshold from 0.7 to 0.6, you'll find it
