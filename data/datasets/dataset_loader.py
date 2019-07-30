# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img




class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None,partial_crop = None):
        self.dataset = dataset
        self.transform = transform
        self.partial_crop = partial_crop
        self.resize = T.Resize([256,128])
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        img = self.resize(img)
        img_partial,region_label = self.partial_crop(img)
        if self.transform is not None:
            img_partial = self.transform(img_partial)
            img = self.transform(img)
        # print("dataset_loader--ImageDateset:img_partial.shape={}, region_label = {}".format(img_partial.shape,region_label))
        # print("image_dataset---------------",img_path,pid)

        return img, pid, camid, img_path, img_partial, region_label
