# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""

from torch.utils.data import Dataset
import torchvision.transforms as T






class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.resize = T.Resize([224,224])
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid  = self.dataset[index]
        img = read_image(img_path)
        img = self.resize(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, pid
