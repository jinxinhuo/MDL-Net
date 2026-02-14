import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
from pathlib import Path
import cv2
import torch
from dataset.transform import xception_default_data_transforms

class read_mask_01_label_data(Dataset):
    def __init__(self, root_dir, data_type='train'):
        self.root = root_dir
        self.data_type = data_type
        self.labels = []

        fake_path = os.path.join(root_dir, '0')
        fake_path = Path(fake_path)
        fake_list = list(fake_path.glob('*.png'))
        images_fake_str = [str(x) for x in fake_list]
        labels_fake_str = [0 for x in fake_list]

        real_path = os.path.join(root_dir, '1')
        real_path = Path(real_path)
        real_list = list(real_path.glob('*.png'))
        images_real_str = [str(x) for x in real_list]
        labels_real_str = [1 for x in real_list]

        self.labels = labels_real_str + labels_fake_str
        images_list_str = images_real_str + images_fake_str
        self.images = images_list_str
        self.transform = self.get_transform(data_type)

        print("number of images is :", len(images_list_str))
        print("number of labels is :", len(self.labels))

    def get_transform(self, data_type):
        """根据数据类型选择变换"""
        if data_type == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),  # 调整图像大小为 256x256 像素
                transforms.RandomHorizontalFlip(p=0.5),  # 以 0.5 的概率随机水平翻转
                transforms.ToTensor(),  # 将图像转换为 Tensor，并将像素值缩放到 [0, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),  # 调整图像大小为 256x256 像素
                transforms.ToTensor(),  # 将图像转换为 Tensor，并将像素值缩放到 [0, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
            ])

    def __getitem__(self, item):
        image_path = self.images[item]

        try:
            # 读取原始图片
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
            if image is None:
                print(f"Warning: Failed to load image {image_path}. Skipping.")
                return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # 只对 image 进行 transform（包含 Normalize）
        image = self.transform(image)

        # 获取标签
        label = 1
        label_file = Path(image_path).parent.name
        if label_file == '1':
            label = 1
        elif label_file == '0':
            label = 0
        else:
            print('There is a wrong label name!')
            exit(0)

        # 定义 mask 的 transform（不进行 Normalize）
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # 处理 mask 和 binary_mask
        mask_image = None
        binary_mask_image = None

        if label == 0:
            mask_image_name = Path(image_path).name
            mask_image_path = os.path.join(self.root, 'mask', mask_image_name)
            binary_mask_image_path = os.path.join(self.root, '01mask', mask_image_name)

            try:
                mask_image = cv2.imdecode(np.fromfile(mask_image_path, dtype=np.uint8), 1)
                if mask_image is None:
                    print(f"Warning: Failed to load mask image {mask_image_path}. Skipping.")
                    return None
                binary_mask_image = cv2.imdecode(np.fromfile(binary_mask_image_path, dtype=np.uint8), 0)  # 读取灰度
                if binary_mask_image is None:
                    print(f"Warning: Failed to load binary mask image {binary_mask_image_path}. Skipping.")
                    return None
            except Exception as e:
                print(f"Error loading mask image {mask_image_path}: {e}. Skipping.")
                return None

            # 处理 mask（转换为 RGB）
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            mask_image = Image.fromarray(mask_image)
            mask_image = mask_transform(mask_image)

            # 处理 binary_mask（**单通道**）
            binary_mask_image = Image.fromarray(binary_mask_image)  # 仍然是灰度图
            binary_mask_image = mask_transform(binary_mask_image)  # 这里应该是 (1, 256, 256)
            # 如果 binary_mask_image 形状是 (1, 256, 256)，我们通过 squeeze 去掉通道维度
            binary_mask_image = binary_mask_image.squeeze(0)  # 变成 (256, 256)

        elif label == 1:
            mask_image = torch.zeros((3, 256, 256))  # mask 仍然是 3 通道
            # binary_mask_image = torch.zeros((1, 256, 256))  # **修正为 1 通道**
            binary_mask_image = torch.zeros((256, 256))  # **修正为 1 通道**

        return image, label, mask_image, binary_mask_image

    def __len__(self):
        return len(self.images)