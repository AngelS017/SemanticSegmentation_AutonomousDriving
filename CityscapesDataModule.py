import torch
import lightning.pytorch as pl
from torchvision import datasets
from torch.utils.data import DataLoader

from utils import preprocesing_utils

class CityscapesDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the Cityscapes dataset.

    Handles downloading, preprocessing, and creating DataLoader instances
    for the train, validation, and test splits.

    Parameters
    ----------
    data_dir : str
        Directory where the dataset is stored or will be downloaded.
    batch_size : int
        Batch size for training and testing. Validation uses a batch size of 1.
    num_workers : int
        Number of workers for the DataLoaders.
    mean : list[float] | torch.Tensor
        Mean values for normalizing the images.
    std : list[float] | torch.Tensor
        Standard deviation values for normalizing the images.
    crop_size : tuple[int, int]
        Spatial size (H, W) for random cropping during training.
    """
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, 
                 mean: list[float] | torch.Tensor, std: list[float] | torch.Tensor, crop_size: tuple[int, int]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.crop_size = crop_size

    def setup(self, stage: str | None = None) -> None:
        """
        Sets up the transforms and creates the PyTorch datasets.

        Parameters
        ----------
        stage : str | None, optional
            The stage of training (fit, test, etc.), by default None.
        """
        transforms = preprocesing_utils.create_transforms(mean=self.mean, std=self.std, crop_size=self.crop_size)
        self.train_transform = transforms['train']
        self.val_test_transform = transforms['val_test']

        self.train_data = datasets.Cityscapes(root=self.data_dir, split='train', mode='fine', target_type='semantic', 
                                              transforms=self.train_transform)
        self.val_data = datasets.Cityscapes(root=self.data_dir, split='val', mode='fine', target_type='semantic',
                                            transforms=self.val_test_transform)
        self.test_data = datasets.Cityscapes(root=self.data_dir, split='test', mode='fine', target_type='semantic',
                                             transforms=self.val_test_transform)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.
        
        Returns
        -------
        DataLoader
            DataLoader for the training split.
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, 
                          num_workers=self.num_workers, prefetch_factor=2, persistent_workers=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.
        
        Returns
        -------
        DataLoader
            DataLoader for the validation split.
        """
        return DataLoader(self.val_data, batch_size=1, shuffle=False, pin_memory=True, 
                          num_workers=self.num_workers, prefetch_factor=2, persistent_workers=True, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.
        
        Returns
        -------
        DataLoader
            DataLoader for the test split.
        """
        return DataLoader(self.test_data, batch_size=1, shuffle=False, pin_memory=True, 
                          num_workers=1, prefetch_factor=1, persistent_workers=True, drop_last=True)