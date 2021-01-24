import os
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

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

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l2'))])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))


        custom_path = os.path.join(self.root, 'custom')
        train_path = os.path.join(custom_path, 'train.csv')
        train_set = MyCustom(root=custom_path, fpath=train_path, transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.target_ten.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        custom_path = os.path.join(self.root, 'custom')
        test_path = os.path.join(custom_path, 'test.csv')
        self.test_set = MyCustom(root=custom_path, fpath=test_path, transform=transform, target_transform=target_transform)

        # Unlabelled data
        self.apply_set = MyCustom(root=custom_path, fpath=train_path, transform=transform,  target_transform=None) # labels are NaN


class MyCustom(Dataset):
    
    def __init__(self, root, fpath, transform=None, target_transform=None):
        super().__init__()

        self.custom_path = root

        # Transforms
        self.to_tensor = transforms.ToTensor()
        
        # Read the csv file
        self.data_info = pd.read_csv(fpath)
        
        # First column contains the image paths
        self.image_arr = np.array(self.data_info.iloc[:, 0])
        
        # Second column is the targets (labels)
        self.target_arr = np.array(self.data_info.iloc[:, 1])
        self.target_ten = torch.from_numpy(self.target_arr)
        
        # Calculate len        
        self.data_len = len(self.data_info.index)
        
        #initializing transforms        
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        try:

            # Get image name
            image_name = os.path.join(self.custom_path, self.image_arr[index])

            # Open image and convert to greyscale
            img = Image.open(image_name).convert('L')

            # Get target (label) of the image
            target = torch.from_numpy(np.array(self.target_arr[index]))

        except:

            # Get image name
            image_name = os.path.join(self.custom_path, self.image_arr[0])

            # Open image and convert to greyscale
            img = Image.open(image_name).convert('L')

            # Get target (label) of the image
            target = torch.from_numpy(np.array(self.target_arr[0]))

        finally:
            # Transform image
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)  

        return img, target, index

    def __len__(self):
        return self.data_len
