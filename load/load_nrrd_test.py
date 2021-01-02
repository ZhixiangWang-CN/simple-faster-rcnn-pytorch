import torch
from torch.utils.data import Dataset, DataLoader
from load.normalization import min_max_normalize
from skimage.transform import resize
import nrrd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
root = ""


# -----------------ready the dataset--------------------------

#
# def nrrd_loader(path):
#     data, header = nrrd.read(path)
#     return data

def read_image_from_nrrd(image_path,hu_min,hu_max):
        image, header = nrrd.read(image_path)
        image = resize(image, (256, 256))

        image = torch.Tensor(image)
        image = torch.clamp(image, hu_min, hu_max)
        # Normalize Hounsfield units to range [-1,1]
        image = min_max_normalize(image, hu_min, hu_max)
        image = image.unsqueeze(0)
        return image

def get_roi_label(image_path):
    names = image_path.replace('.nrrd', '')
    names = names.split('_')
    print("name", names)
    roi = []
    coefficient = 256 / 512
    mask = np.zeros([256, 256])
    for x in names[3:-1]:
        roi.append(int(int(x) * coefficient))
    # labels_roi_type.append(int(names[-1]))
    # print(labels_roi_type)
    # print("roi",roi)
    mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1
    return mask
def deal_image(images):
    images = images.to('cpu')
    images = images.detach().numpy()
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    return images

if __name__ =='__main__':
    img_path = '../tools/lung/erased'
    hu_min, hu_max = -1000, 2000
    for file in os.listdir(img_path):
        fn = os.path.join(img_path, file)
        fn = fn.replace("\\", '/')


        fn_label = fn.replace('erased', 'label')
        image = read_image_from_nrrd(fn, hu_min, hu_max)
        label_image = read_image_from_nrrd(fn_label, hu_min, hu_max)
        mask = get_roi_label(fn)
        out_image = label_image.clone()
        out_image[0]=out_image[0]*mask
        image = deal_image(image)
        out_image=deal_image(out_image)
        plt.subplot(1,3,1)
        plt.imshow(image[0])
        plt.subplot(1, 3, 2)
        plt.imshow(out_image[0])
        plt.subplot(1, 3, 3)
        plt.imshow(label_image[0])
        plt.show()





