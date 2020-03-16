import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import cv2

##导入albumentations来做图像增强
input_size = 224
import albumentations
from albumentations.augmentations.transforms import Resize, RandomSizedCrop, ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, MotionBlur, Blur, GaussNoise, JpegCompression

train_transform = albumentations.Compose([
                                          RandomSizedCrop(min_max_height=(input_size//3,input_size//3),height=input_size,width=input_size),
                                          ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1, rotate_limit=25),
                                          HorizontalFlip(p=0.2),
                                          RandomBrightnessContrast(p=0.3, brightness_limit=0.25, contrast_limit=0.5),
                                          MotionBlur(p=.2),
                                          GaussNoise(p=.2),
                                          JpegCompression(p=.2, quality_lower=50),
                                          Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])
val_transform = albumentations.Compose([
                                        Resize(input_size,input_size),
                                        Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])

class MyDataset(data.Dataset):
    def __init__(self,root,transforms = None,is_train=True):
        """
        主要目标获取所有图片的地址
        """
        fake_path = os.path.join(root,"fake")
        real_path = os.path.join(root,"real")

        fake_imgs = [os.path.join(fake_path,each) for each in os.listdir(fake_path) if each.endswith(".jpg") or each.endswith(".png") or each.endswith(".jpeg") or each.endswith(".JPG")]
        real_imgs = [os.path.join(real_path,each) for each in os.listdir(real_path) if each.endswith(".jpg") or each.endswith(".png") or each.endswith(".jpeg") or each.endswith(".JPG")]
        imgs = fake_imgs+real_imgs
        self.imgs = imgs
        self.transforms = transforms

    def __getitem__(self,index):
        """
        一次返回一张图像
        """
        img_path = self.imgs[index]
        print(img_path)
        label = 0 if img_path.split("/")[-2]=="fake" else 1
        print(label)
        data = Image.open(img_path)
        data = np.array(data)
        if self.transforms is not None:
            res = self.transforms(image = data)
            data = res['image']
        return data,label

    def __len__(self):
        return len(self.imgs)
