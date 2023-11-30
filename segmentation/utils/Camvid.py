# 导入库
import os
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

torch.manual_seed(17)


# 自定义数据集CamVidDataset
class CamVidDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """

    def __init__(self, images_dir, masks_dir):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('P'))
        image = self.transform(image=image, mask=mask)

        return image['image'], image['mask']

    def __len__(self):
        return len(self.ids)


class Bi_Dataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    def __init__(self, pre_dir, post_dir, masks_dir):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.ids = os.listdir(pre_dir)
        self.pre_fps = [os.path.join(pre_dir, image_id) for image_id in self.ids]
        self.post_fps = [os.path.join(post_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

    def __getitem__(self, i):
        # read data
        pre = np.array(Image.open(self.pre_fps[i]).convert('RGB'))
        post = np.array(Image.open(self.post_fps[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('P'))
        pre = self.transform(image=pre, mask=mask)
        post = self.transform(image=post, mask=mask)

        return pre['image'], post['image'], post['mask']

    def __len__(self):
        return len(self.ids)
