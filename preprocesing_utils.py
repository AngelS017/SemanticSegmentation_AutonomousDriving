import os
from zipfile import ZipFile
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
cv2.setNumThreads(0) # Disable OpenCV thread pool to prevent OOM in Dataloaders


# We need to transform the ID of each pixel in the target mask to the Train ID, due to some of the classes are ignored
# with this we pass from 30 classes to the 19 used for training and evaluatin the models.
CITYSCAPES_MAPPING_ID_FOR_TRAIN = torch.tensor([
    255, 255, 255, 255, 255, 255, 255, 0,   1,   255, 
    255, 2,   3,   4,   255, 255, 255, 5,   255, 6, 
    7,   8,   9,   10,  11,  12,  13,  14,  15,  255, 
    255, 16,  17,  18
], dtype=torch.long)


def download_dataset(data_path, name_file_zip, url_dataset):
    """
    Downloads a dataset from a specified URL and saves it to a local file.

    Parameters
    ----------
    data_path : str
        The local directory path where the dataset will be saved.   
    name_file_zip : str
        The name of the zip file to save the dataset as.
    url_dataset : str
        The URL from which to download the dataset.

    Returns
    ------
    None
    
    """

    if os.path.exists(data_path) and os.path.isdir(data_path):
        print("DataSet folder found locally. Loading from local.\n")
    else:
        print("DataSet folder not found locally. Downloading from Kaggle.\n")
        # Create data directory if it does not exist
        os.makedirs(data_path, exist_ok=True)

        # Download dataset from Kaggle using curl
        download_status = os.system(f'curl -L -o "{os.path.join(data_path, name_file_zip)}" {url_dataset}')
        print()

        if download_status == 0:
            print("Download completed. Extracting files...\n")
            with ZipFile(os.path.join(data_path, name_file_zip), 'r') as zip_ref:
                zip_ref.extractall(data_path)
            print("Extraction completed.\n")
        else:
            print("Error downloading dataset from Kaggle. Please check your internet connection and Kaggle API credentials.\n")
            exit(1)


def calculate_mean_std(train_data):
    """
    Calculate the mean and standard deviation of the dataset.
    Using the formula:
        std = sqrt(E[x^2] - (E[x])^2)

    Parameters
    ----------
    train_data : Dataset
        The training dataset.

    Returns
    -------
    mean_channels : torch.Tensor
        The mean for each channel.
    
    std_channels : torch.Tensor
        The standard deviation for each channel.

    """
    dataloader = DataLoader(train_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=2)

    num_pixels = 0
    sum_channels = torch.zeros(3)
    sum_squared_channels = torch.zeros(3)

    for images, _ in tqdm(dataloader, desc="Calculating Dataset Stats"):
        # Get the number of pixels in the batch for each channel
        num_pixels += images.size(0) * images.size(2) * images.size(3)
        # Sum the pixel values for each channel
        sum_channels += images.sum(dim=[0, 2, 3])
        # Sum the squared pixel values for each channel
        sum_squared_channels += (images ** 2).sum(dim=[0, 2, 3])

    mean_channels = sum_channels / num_pixels
    std_channels = torch.sqrt((sum_squared_channels / num_pixels) - (mean_channels ** 2))
    
    return mean_channels, std_channels


def prepare_images_targets_tensors(images, targets):
    """
    Prepares the images and target masks by converting to tensors and set if they are Images or Masks, in order to
    apply some of the transformations only to the images and not to the target masks, such as the color jitter or the Gaussian blur.
    
    Parameters
    ----------
    images : list or PIL.Image
        A batch of input images, typically in the form of a list of PIL.Image objects or a single PIL.Image object.
    targets : list or PIL.Image
        A batch of target masks corresponding to the input images, typically in the form of a list

    Returns
    -------
    tuple
        A tuple containing the prepared images and target masks as tensors, ready for further transformations.
    """
    images = np.array(images)
    targets = np.array(targets)

    return images, targets


def format_target_mask_tensor(images, targets):
    """
    Formats the target mask tensor by converting it to a long tensor and removing the channel dimension.

    Parameters
    ----------
    target_mask : torch.Tensor
        A tensor representing the target mask, typically with shape (B, 1, H, W) where B is the batch size.

    Returns
    -------
    torch.Tensor
        A formatted target mask tensor with shape (B, H, W) and dtype torch.long, suitable for use in loss functions like CrossEntropyLoss.
    """
    targets = targets.to(torch.long)

    mappsing_id_to_train_id = CITYSCAPES_MAPPING_ID_FOR_TRAIN.to(targets.device)
    targets = mappsing_id_to_train_id[targets]

    return images, targets


class AlbumentationsCityscapesWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, targets):
        images, targets = prepare_images_targets_tensors(images, targets)
        transformed = self.transform(image=images, mask=targets)
        images = transformed['image']
        targets = transformed['mask']
        images, targets = format_target_mask_tensor(images, targets)
        return images, targets


def create_transforms(mean, std, crop_size):
    """
    Creates data augmentation and normalization transforms for training, validation, and testing.

    Parameters
    ----------
    mean : list or torch.Tensor
        The mean values for each channel to be used in normalization.
    std : list or torch.Tensor
        The standard deviation values for each channel to be used in normalization.

    Returns
    -------
    dict
        A dictionary containing the transforms for training, validation/testing, and target transformations.
    """ 

    """
    blur_or_noise = A.Compose([
        A.RandomApply([A.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 5.))], p=0.4),
        A.RandomApply([A.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.4)
    ])
    """

    blur_or_noise = A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(9, 9), sigma_limit=(0.1, 5.0), p=0.4),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
        ], p=1.0)
    ])  

    """
    train_transform = v2.Compose([
        prepare_images_targets_tensors,
        
        # Transform firts the PIL image to tensors to maximize performance of the data augmentations with the v2 API
        v2.ToImage(),

        # Realize a RandomCrop with a smaller size to reduce training memory and avoid overfitting
        v2.RandomCrop(size=(512, 1024)),
        v2.RandomHorizontalFlip(p=0.4),
        v2.RandomRotation(degrees=5, fill=0), # Fill with 0 to ignore them in the loss
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        blur_or_noise,

        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),

        format_target_mask_tensor
    ])
    """

    train_transform_compose = A.Compose([
        A.RandomCrop(height=crop_size[0], width=crop_size[1]),
        A.HorizontalFlip(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.4),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        blur_or_noise,

        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

    """
    val_test_transform = v2.Compose([
        prepare_images_targets_tensors,

        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),

        format_target_mask_tensor
    ])
    """

    val_test_transform_compose = A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

    return {
        "train": AlbumentationsCityscapesWrapper(train_transform_compose),
        "val_test": AlbumentationsCityscapesWrapper(val_test_transform_compose)
    }


def denormalize_image(image, mean, std):
    """
    Denormalizes an image tensor using the provided mean and standard deviation.

    Parameters
    ----------
    image : torch.Tensor
        The normalized image tensor to be denormalized.
    mean : orch.Tensor
        The mean values for each channel used in normalization.
    std : torch.Tensor
        The standard deviation values for each channel used in normalization.

    Returns
    -------
    torch.Tensor
        The denormalized image tensor.
    """
    # Reformat mean and std to match the image tensor shape (C, H, W) for broadcasting with batch size
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    denormalized_image = (image * std) + mean

    return denormalized_image


def display_grid(grid_images):
    """
    Displays a grid of images using matplotlib, clipping values to the valid range.

    Parameters
    -------
    grid (torch.Tensor): A grid of images created by vutils.make_grid.

    Returns
    -------
    None
    
    """
    # Convert tensor to NumPy array and transpose from (C, H, W) to (H, W, C)
    #grid_images = grid_images.detach().cpu()
    grid_np = np.transpose(grid_images.numpy(), (1, 2, 0))

    # Clip the data to the valid display range [0, 1] for floats
    clipped_grid = np.clip(grid_np, 0, 1)

    # Display the clipped image
    plt.figure(figsize=(20, 20))
    plt.imshow(clipped_grid)
    plt.axis('off')
    plt.show()


def colored_target_mask(target_mask, info_classes):
    """
    Decodes a target mask by mapping class indices to their corresponding RGB colors.

    Parameters
    ----------
    target_mask : torch.Tensor
        A tensor containing class indices for each pixel in the target mask.
    info_classes : dict
        A dictionary mapping class indices to their corresponding RGB color values.

    Returns
    -------
    torch.Tensor
        A tensor representing the decoded target mask with RGB color values.
    """
    batch_size, hight, width = target_mask.shape
    decoded_mask = torch.zeros((batch_size, 3, hight, width), dtype=torch.uint8)

    for class_dataset in info_classes:
        mask_class = (target_mask == class_dataset.train_id)
        decoded_mask[:,0,:,:][mask_class] = class_dataset.color[0]
        decoded_mask[:,1,:,:][mask_class] = class_dataset.color[1]
        decoded_mask[:,2,:,:][mask_class] = class_dataset.color[2]
        
    return decoded_mask.float() / 255.0


def calculate_class_weights(train_dataloader, num_classes=19, method='enet', c=1.02, ignore_index=255):
    """
    Calculate class weights for a given dataset to handle class imbalance by counting pixel frequencies.

    Parameters
    ----------
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for the training dataset.
    num_classes : int, optional
        The number of classes in the dataset (default is 19 for Cityscapes training).
    method : str, optional
        The method to use for calculating weights: 'inverse', 'median', or 'enet' (default is 'enet').
    c : float, optional
        The constant used in the ENet method (default is 1.02).
    ignore_index : int, optional
        The index to ignore when calculating weights (e.g., 255 for unlabeled pixels).

    Returns
    -------
    torch.Tensor
        A tensor containing the calculated weights for each class.
    """
    # Initialize counts for each class
    class_counts = torch.zeros(num_classes)
    
    # Iterate through the dataloader to count pixels for each class
    for _, targets in tqdm(train_dataloader, desc=f"Calculating class weights ({method})"):
        targets = targets.view(-1)
        # Filter out the ignore_index
        mask = (targets != ignore_index)
        valid_targets = targets[mask]
        
        # Increment counts using bincount for efficiency
        if valid_targets.numel() > 0:
            class_counts += torch.bincount(valid_targets, minlength=num_classes).float()

    # Calculate pixel frequencies
    total_pixels = class_counts.sum()
    frequencies = class_counts / total_pixels

    # Apply the selected weighting method
    if method == 'inverse':
        # Standard inverse frequency weighting
        weights = 1.0 / (frequencies + 1e-6)
        # Normalize weights to achive mean of 1, usefull for the weighted loss 
        # function and the learning rate stay in a good range
        weights = weights / weights.sum() * num_classes
    elif method == 'median':
        # Median frequency balancing: median(frequencies) / frequencies[i]
        median_freq = torch.median(frequencies)
        weights = median_freq / (frequencies + 1e-6)
        # Normalize
        weights = weights / weights.sum() * num_classes
    elif method == 'enet':
        # ENet-style logarithmic balancing: 1 / ln(c + frequency)
        # This method is less sensitive to extremely rare classes
        weights = 1.0 / torch.log(c + frequencies)
        # Normalize
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Method '{method}' not recognized. Use 'inverse', 'median', or 'enet'.")

    return weights
