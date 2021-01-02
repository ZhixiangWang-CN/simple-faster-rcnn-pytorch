import torch
from torch.utils.data import Dataset, DataLoader
from load.normalization import min_max_normalize
from skimage.transform import resize
import nrrd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

root = ""


# -----------------ready the dataset--------------------------


def read_image_from_nrrd(image_path, hu_min, hu_max):
    image, header = nrrd.read(image_path)
    # image = resize(image, (3, 256, 256))
    image = np.transpose(image,[2,0,1])
    # print("image size",image.shape)
    # plt.imshow(image[1,:,:])
    # plt.show()
    image = torch.Tensor(image)
    image = torch.clamp(image, hu_min, hu_max)
    # Normalize Hounsfield units to range [-1,1]
    image = min_max_normalize(image, hu_min, hu_max)
    # image = image.unsqueeze(0)
    return image


def get_roi_label(image_path):
    names = image_path.replace('.nrrd', '')
    names = names.split('_')
    # print("name", names)
    roi = []
    # label_type = []
    coefficient = 512 / 512
    mask = np.zeros([256, 256])
    for x in names[3:7]:
        roi.append(int(int(x) * coefficient))
    # label_type.append(int(names[-2])-1)
    label_type=int(names[-1])-1
    # label_type.append(int(names[-1])-1)
    # labels_roi_type.append(int(names[-1]))
    # print(labels_roi_type)
    # print("roi",roi)
    # mask[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]=1

    return roi, label_type


def cut_roi(image, roi):
    w = 32
    middle_point=[roi[0]+roi[2]//2,roi[1]+roi[3]//2]
    # print("middle",middle_point)
    cut = image[:,middle_point[1]-w:middle_point[1]+w,middle_point[0]-w:middle_point[0]+w]
    # print("cut",cut.shape)
    return cut


class MyDataset(Dataset):
    # 构造函数带有默认参数

    def __init__(self, path):
        self.img_path = path

        self.img_list = []

        for file in os.listdir(self.img_path):
            img_path = os.path.join(self.img_path, file)
            img_path = img_path.replace("\\", '/')
            # print(img_path)
            self.img_list.append(img_path)

        self.hu_min, self.hu_max = -1000, 2000

    def __getitem__(self, index):
        fn = self.img_list[index]

        fn_label = fn.replace('erased', 'label')
        image = read_image_from_nrrd(fn, self.hu_min, self.hu_max)
        label_image = read_image_from_nrrd(fn_label, self.hu_min, self.hu_max)

        roi, label_type = get_roi_label(fn)
        # print(roi)
        image_cut = cut_roi(image, roi)
        label_image_cut = cut_roi(label_image, roi)
        # print("image_cut",image_cut[1:2,:,:].shape)
        # plt.subplot(2,2,1)
        # plt.imshow(image[1]*255)
        # plt.subplot(2, 2, 2)
        # plt.imshow(label_image[1]*255)
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(image_cut[1] * 255)
        # plt.subplot(2, 2, 4)
        # plt.imshow(label_image_cut[1] * 255)
        # plt.show()
        return image_cut[1:2,:,:], label_image_cut[1:2,:,:], roi, label_type

    def __len__(self):
        return len(self.img_list)

if __name__ =='__main__':
    img_path = '/workspace/data/lungnrrd/erased'
    train_data = MyDataset(path=img_path)
    test_data = MyDataset(path=img_path)

    # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=5, shuffle=True)

    test_loader = DataLoader(dataset=test_data, batch_size=2)

    for x,x_,y,label in train_loader:
        # print(x.shape)
        # print(x_.shape)
        # # # print("x",x)
        # print("y", y)
        print(label)
        one=torch.nn.functional.one_hot(label,num_classes=5)
        print(one)