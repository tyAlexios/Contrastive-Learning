import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class CUHK01(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.num_pos = 4  # the number of positive pairs
        self.data_info = self.get_img_label(root, self.num_pos, train)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_label(root, num_pos, train):
        data_info = list()
        cur_label = 0

        if train:
            sub_dir = 'train'
        else:
            sub_dir = 'test'
            
        img_names = os.listdir(os.path.join(root, sub_dir))
        img_names = sorted(list(filter(lambda x: x.endswith('.png'), img_names)))

        for i in range(0, len(img_names), num_pos):
            for j in range(i, i + num_pos):
                img_name = img_names[j]
                path_img = os.path.join(root, sub_dir, img_name)
                label = cur_label
                data_info.append((path_img, int(label)))
            cur_label += 1

        return data_info




