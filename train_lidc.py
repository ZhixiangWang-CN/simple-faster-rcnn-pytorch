from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config_LIDC import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from load.load_nrrd import MyDataset
from torch.utils.data import DataLoader
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def train(**kwargs):
    opt._parse(kwargs)
    # print(opt)
    num_img =opt.num_workers
    img_path = opt.voc_data_dir
    # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
    train_data = MyDataset(img_path)
    test_data = MyDataset(img_path)
    # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=num_img, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    

    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img,label_img,bbox_, label_) in tqdm(enumerate(train_loader)):
            label_=label_.unsqueeze(0)
            scale = 1.0
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)
            
        
        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()
