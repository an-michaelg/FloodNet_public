import os
import glob
import random
import functools

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image

def seed_everything(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)

seed_everything(42)

def get_dataloader(config, mode='train'):
    img_dim = config.image_dim
    dataset = DataLoaderSegmentation('./dataset/train/Labeled', img_dim)
    loader = functools.partial(
        DataLoader,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    loader_list = []
    if mode=='train':
        ratio_tr_data = config.ratio_tr_data
        num_all = len(dataset)

        idx = np.random.permutation(np.arange(num_all))
        num_tr = int(ratio_tr_data * num_all)

        dataset_tr = Subset(dataset, idx[:num_tr])
        dataset_va = Subset(dataset, idx[num_tr:])

        loader_tr = loader(dataset=dataset_tr, shuffle=True)
        loader_va = loader(dataset=dataset_va, shuffle=False)

        loader_list += [loader_tr, loader_va]
        print(f"Number of training samples: {num_tr}")
        print(f"Number of valid samples: {num_all - num_tr}")
    else:
        raise NotImplementedError
    return loader_list

class DataLoaderSegmentation(Dataset):
    def __init__(self, img_folder, img_dim):
        """ img_dim is a tuple of (h,w) """
        self.img_files = glob.glob(os.path.join(img_folder,'**','*.jpg'), recursive=True)
        self.mask_files = glob.glob(os.path.join(img_folder,'**','*.png'), recursive=True)
        self.h = img_dim[0]
        self.w = img_dim[1]
        assert len(self.img_files) == len(self.mask_files) # consistency check

    def __getitem__(self, index):
        im = Image.open(self.img_files[index])
        mask = Image.open(self.mask_files[index])
        return self._transform(im, mask)

    def __len__(self):
        return len(self.img_files)
    
    def _transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(self.h, self.w))
        image = resize(image)
        mask = resize(mask)

#        # Random crop
#        i, j, h, w = transforms.RandomCrop.get_params(
#            image, output_size=(self.h, self.w))
#        image = TF.crop(image, i, j, h, w)
#        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)
        
        # Normalize image (with imageNet mean/stdev)
        image = TF.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        data = {
            "image": image,
            "mask": mask
            }
        return data


# from utils import vis_pair
# dset = DataLoaderSegmentation('./dataset/train/Labeled', (400, 600))
# loader = DataLoader(dataset=dset, shuffle=True, batch_size=1, num_workers=0)
# data = next(iter(loader)) 
# vis_pair(data['image'][0], data['mask'][0])
# mask = data[1]
# import torch.nn.functional as F
# mask_oh = F.one_hot(mask, num_classes=10)#, num_classes=10)