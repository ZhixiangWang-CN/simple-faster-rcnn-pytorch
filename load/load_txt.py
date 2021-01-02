import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from load.normalization import min_max_normalize
from torchvision import transforms
from skimage.transform import resize
root = ""


# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')

def npy_loader(path):
    image = np.load(path)
    return image

class MyDataset(Dataset):
    # 构造函数带有默认参数
    def __init__(self, txt, transform=None, target_transform=None, loader=npy_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            # 移除字符串首尾的换行符
            # 删除末尾空
            # 以空格为分隔符 将字符串分成
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            label_temp = int(words[-1])
            imgs.append((words[0], label_temp))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.hu_min, self.hu_max=-1000,2000

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        fn_label = fn.replace('erased','cut')
        # 调用定义的loader方法
        img = self.loader(fn)
        img_label = self.loader(fn_label)
        img = resize(img, (256, 256))
        img_label = resize(img_label, (256, 256))

        # print(image.shape)
        img = torch.Tensor(img)
        img_label = torch.Tensor(img_label)
        img = torch.clamp(img, self.hu_min, self.hu_max)
        img_label = torch.clamp(img_label, self.hu_min, self.hu_max)

        # Normalize Hounsfield units to range [-1,1]
        img = min_max_normalize(img, self.hu_min, self.hu_max)
        img_label = min_max_normalize(img_label, self.hu_min, self.hu_max)
        img=img.unsqueeze(0)
        img_label=img_label.unsqueeze(0)
        return img,img_label, label-1
    def __len__(self):
        return len(self.imgs)


if __name__ =='__main__':
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
])
    train_data = MyDataset(txt=root + 'Erased.txt', transform=data_transforms)
    test_data = MyDataset(txt=root + 'Erased.txt', transform=data_transforms)

    # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=2)

    for x,x_,y in train_loader:
        print(x.shape)
        # print("x",x)
        # print("y", y)