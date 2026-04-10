import lightning.pytorch as pl
from torchvision import datasets
from torch.utils.data import DataLoader

import preprocesing_utils

class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, mean, std, crop_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.crop_size = crop_size

    def setup(self, stage=None):
        transforms = preprocesing_utils.create_transforms(mean=self.mean, std=self.std, crop_size=self.crop_size)
        self.train_transform = transforms['train']
        self.val_test_transform = transforms['val_test']

        self.train_data = datasets.Cityscapes(root=self.data_dir, split='train', mode='fine', target_type='semantic', 
                                              transforms=self.train_transform)
        self.val_data = datasets.Cityscapes(root=self.data_dir, split='val', mode='fine', target_type='semantic',
                                            transforms=self.val_test_transform)
        self.test_data = datasets.Cityscapes(root=self.data_dir, split='test', mode='fine', target_type='semantic',
                                             transforms=self.val_test_transform)

    def train_dataloader(self):
        # Return the training dataloader
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, 
                          num_workers=self.num_workers, prefetch_factor=2, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(self.val_data, batch_size=1, shuffle=False, pin_memory=True, 
                          num_workers=self.num_workers, prefetch_factor=2, persistent_workers=True, drop_last=True)

    def test_dataloader(self):
        # Return the test dataloader
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, 
                          num_workers=self.num_workers, prefetch_factor=2, persistent_workers=True, drop_last=True)