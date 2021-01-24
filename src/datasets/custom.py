import os
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

class Custom_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class :int):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 1))

        self.train_data = []
        self.train_labels = []

        self.test_data = []
        self.test_labels = []


        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                             [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))


        custom_path = os.path.join(self.root, 'custom')
        train_path = os.path.join(custom_path, 'train.csv')
        train_set = MyCustom(root=custom_path, fpath=train_path, train=True, transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        # train_idx_normal = get_target_label_idx(train_set.target_ten.clone().data.cpu().numpy(), self.normal_classes)
        # self.train_set = Subset(train_set, train_idx_normal)
        self.train_set = Subset(train_set, train_set.target_idx)

        custom_path = os.path.join(self.root, 'custom')
        test_path = os.path.join(custom_path, 'test.csv')
        self.test_set = MyCustom(root=custom_path, fpath=test_path, train=False, transform=transform, target_transform=target_transform)

        # Unlabelled data
        self.apply_set = MyCustom(root=custom_path, fpath=train_path, train=False, transform=transform,  target_transform=None) # labels are NaN


class MyCustom(Dataset):
    
    def __init__(self, root, fpath, train, transform=None, target_transform=None):
        super().__init__()

        self.custom_path = root

        # Transforms
        self.to_tensor = transforms.ToTensor()
        
        # Read the csv file
        self.data_info = pd.read_csv(fpath)
        
        # First column contains the image paths
        image_arr = []
        for i,r in self.data_info.iterrows():
            tmp_img = Image.open(os.path.join(self.custom_path, r[0]))
            tmp_img = ImageOps.grayscale(tmp_img)
            image_arr.append(np.array(tmp_img))
        
        # Second column is the targets (labels)
        target_arr = np.array(self.data_info.iloc[:, 1])
        # self.target_arr = self.data_info[1]
        # self.target_ten = torch.from_numpy(target_arr)
        self.target_idx = np.array(self.data_info.index)
        
        # Calculate len        
        self.data_len = len(self.data_info.index)
        
        #initializing transforms        
        self.transform = transform
        self.target_transform = target_transform

        self.train = train

        if self.train:
            self.train_data = image_arr
            self.train_labels = target_arr
        else:
            self.test_data = image_arr
            self.test_labels = target_arr


    def __getitem__(self, index):

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # try:

        #     # Get image name
        #     image_name = os.path.join(self.custom_path, self.image_arr[index])

        #     # Open image and convert to greyscale
        #     img = Image.open(image_name).convert('L')

        #     # Get target (label) of the image
        #     target = torch.from_numpy(np.array(self.target_arr[index]))

        # except:

        #     # Get image name
        #     image_name = os.path.join(self.custom_path, self.image_arr[0])

        #     # Open image and convert to greyscale
        #     img = Image.open(image_name).convert('L')

        #     # Get target (label) of the image
        #     target = torch.from_numpy(np.array(self.target_arr[0]))

        # finally:
        #     # Transform image
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     if self.target_transform is not None:
        #         target = self.target_transform(target)  

        return img, target, index

    def __len__(self):
        return self.data_len
