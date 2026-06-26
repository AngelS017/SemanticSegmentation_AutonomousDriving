import lightning.pytorch as pl
import torch
from torch import optim
from torchmetrics.classification import MulticlassJaccardIndex
from typing import Any

class DriveSegmentationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training a semantic segmentation model.

    Parameters
    ----------
    model : torch.nn.Module
        The semantic segmentation model (e.g., U-Net).
    loss : torch.nn.Module
        The loss function used for training.
    batch_size : int
        Batch size used for training and validation.
    enc_learning_rate : float
        Learning rate for the encoder.
    dec_learning_rate : float
        Learning rate for the decoder.
    max_epochs : int
        Maximum number of training epochs.
    crop_size : tuple[int, int]
        Size of the image crops used during training and validation.
    stride : tuple[int, int]
        Stride used for the overlap-tile strategy during validation.
    weight_decay : float, optional
        Weight decay (L2 penalty) for the optimizer, by default 1e-2.
    warmup_epochs : int, optional
        Number of epochs for the linear learning rate warmup, by default 5.
    unfreeze_epoch : int | None, optional
        Epoch number at which to unfreeze the encoder layers, by default 10.
    """
    def __init__(self, model: torch.nn.Module, loss: torch.nn.Module, batch_size: int = 8, enc_learning_rate: float = 1e-5, dec_learning_rate: float = 1e-3, 
                 max_epochs: int = 300, crop_size: tuple[int, int] = (512, 512), stride: tuple[int, int] = (256, 256), weight_decay: float = 1e-5,
                 warmup_epochs: int = 5, unfreeze_epoch: int | None = None):
        super().__init__()

        # Save the hyperparameters passed to the constructor, access via `self.hparams`
        # We don't want to save the model and loss function as hyperparameters
        self.save_hyperparameters(ignore=['model', 'loss'])

        self.model = model.to(memory_format=torch.channels_last)
        self.model = torch.compile(self.model)
        self.loss = loss
        self.original_image_size = (1024, 2048)

        # Set other metrics to track
        # The mean IoU metric is calculated as the average of the IoU for each class.
        self.val_mean_iou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255, average='macro', validate_args=False)
        self.train_mean_iou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255, average='macro', validate_args=False)

        # We can save the auxiliary tensor of ones and the count map to reconstruct the image from the tiles because
        # since the size of the original image, crop size, and stride are always the same, we can precompute these tensors.
        # Calculate the number of tiles
        num_tiles = ((self.original_image_size[0] - self.hparams.crop_size[0]) // self.hparams.stride[0] + 1) * \
                    ((self.original_image_size[1] - self.hparams.crop_size[1]) // self.hparams.stride[1] + 1)

        # Create a Gaussian window (patch) with the center value being 1.0
        h, w = self.hparams.crop_size
        
        # 1D Gaussians: exp(-x^2 / (2*sigma^2)). With sigma=0.5, 2*sigma^2 = 0.5
        gauss_h = torch.exp(-torch.linspace(-1, 1, h)**2 / 0.5)
        gauss_w = torch.exp(-torch.linspace(-1, 1, w)**2 / 0.5)
        
        # 2D Gaussian is the outer product of the two 1D Gaussians
        gaussian_patch = gauss_h.unsqueeze(1) * gauss_w.unsqueeze(0)
        
        # Save the Gaussian patch as a buffer to weight predictions during reconstruction
        self.register_buffer("gaussian_patch", gaussian_patch.view(1, 1, h, w))
        
        # Expand the Gaussian patch to simulate the tiles' contribution
        gaussian_aux = gaussian_patch.reshape(1, h * w, 1).expand(1, h * w, num_tiles)
        
        # Accumulate the overlapping Gaussian patches to create the count_map
        count_map = torch.nn.functional.fold(input=gaussian_aux,
                                             output_size=self.original_image_size,
                                             kernel_size=self.hparams.crop_size,
                                             stride=self.hparams.stride)
        
        # Register the count map as a buffer. 
        # Using a small epsilon clamp to prevent dividing by 0 at any border edge.
        self.register_buffer("count_map", count_map.clamp(min=1e-5))

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int | None = None) -> torch.Tensor:
        """
        Defines a single step of training.
        
        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A batch of data from DataLoader, typically a tuple of (inputs, targets).
        batch_idx : int | None
            The index of the current batch. Lightning Trainer requires this argument.
        
        """
        images, targets = batch
        images = images.to(memory_format=torch.channels_last)

        outputs_logits = self.forward(images)
        
        # Calculate the loss
        loss = self.loss(outputs_logits, targets)
        # Log the training loss and iou metric at the end of each epoch
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log the train iou metric
        with torch.no_grad():
            class_prediction = torch.argmax(outputs_logits, dim=1)
            self.train_mean_iou_metric.update(class_prediction, targets)
        self.log('train_iou', self.train_mean_iou_metric, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int | None = None) -> None:
        """
        Defines a single step of validation.  
        
        Parameters        
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A batch of data from DataLoader, typically a tuple of (inputs, targets).
        batch_idx : int | None
            The index of the current batch. Lightning Trainer requires this argument.
        """
        images, targets = batch
        # Apply overlap tile to images
        image_tiles = self._overlap_tile(images, tile_size=self.hparams.crop_size, stride=self.hparams.stride)
        image_tiles = image_tiles.to(memory_format=torch.channels_last)
        
        # Collect tile predictions into a list and cat at the end.
        val_batch_size = self.hparams.batch_size
        tile_outputs = []
        for i in range(0, image_tiles.size(0), val_batch_size):
            chunk = image_tiles[i:i+val_batch_size]
            tile_outputs.append(self.forward(chunk))

        outputs_logits = torch.cat(tile_outputs, dim=0)

        # Reconstruct the full image from tiles
        outputs_logits_reconstructed = self._reconstruct_tile(outputs_logits, tile_size=self.hparams.crop_size, stride=self.hparams.stride, 
                                                              original_size=self.original_image_size)
                                                              
        # Calculate the loss on the full reconstructed image
        loss = self.loss(outputs_logits_reconstructed, targets)
        
        class_prediction = torch.argmax(outputs_logits_reconstructed, dim=1)
        # Calculate the IoU metric on the full reconstructed image.
        self.val_mean_iou_metric.update(class_prediction, targets)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.val_mean_iou_metric, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int | None = None) -> None:
        """
        Defines a single step of testing.
        
        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A batch of data from DataLoader, typically a tuple of (inputs, targets).
        batch_idx : int | None
            The index of the current batch. Lightning Trainer requires this argument.
        """
        images, targets = batch
        #images = images.to(memory_format=torch.channels_last, non_blocking=True)

        outputs_logits = self.forward(images)

        class_prediction = torch.argmax(outputs_logits, dim=1)
        
        # Calculate the loss
        loss = self.loss(outputs_logits, targets)
        # Calculate the IoU metric
        self.mean_iou_metric(class_prediction, targets)
        # Log the test loss and iou metric at the end of each epoch
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_iou', self.mean_iou_metric, on_step=False, on_epoch=True)

    @torch.no_grad()
    def evaluate_model(self, dataloader: torch.utils.data.DataLoader, device: str | torch.device = 'cuda') -> dict[str, float]:
        """
        Evaluates the model on a given dataloader, calculating the loss, mean IoU, and per-class IoU.
        Uses overlap tiling to process high-resolution images correctly.
        """
        self.eval()
        self.to(device)
        
        # Initialize metrics locally
        mean_iou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255, average='macro').to(device)
        class_iou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255, average='none').to(device)
        
        # Tensor to accumulate pixel counts for each class to analyze imbalance
        pixel_counts = torch.zeros(19, dtype=torch.long, device=device)
        
        total_loss = 0.0
        num_batches = 0
        
        from tqdm.auto import tqdm
        
        for batch in tqdm(dataloader, desc="Evaluating Model"):
            images, targets = batch
            images = images.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            
            # Apply overlap tile to images
            image_tiles = self._overlap_tile(images, tile_size=self.hparams.crop_size, stride=self.hparams.stride)
            image_tiles = image_tiles.to(memory_format=torch.channels_last)
            
            # Collect tile predictions into a list and cat at the end.
            val_batch_size = self.hparams.batch_size
            tile_outputs = []
            for i in range(0, image_tiles.size(0), val_batch_size):
                chunk = image_tiles[i:i+val_batch_size]
                tile_outputs.append(self.forward(chunk))
    
            outputs_logits = torch.cat(tile_outputs, dim=0)
    
            # Reconstruct the full image from tiles
            outputs_logits_reconstructed = self._reconstruct_tile(outputs_logits, tile_size=self.hparams.crop_size, stride=self.hparams.stride, 
                                                                  original_size=self.original_image_size)
                                                                  
            # Calculate the loss on the full reconstructed image
            loss = self.loss(outputs_logits_reconstructed, targets)
            total_loss += loss.item()
            num_batches += 1
            
            class_prediction = torch.argmax(outputs_logits_reconstructed, dim=1)
            
            mean_iou_metric.update(class_prediction, targets)
            class_iou_metric.update(class_prediction, targets)
            
            # Count pixels for class frequency analysis (ignoring index 255)
            valid_mask = targets != 255
            pixel_counts += torch.bincount(targets[valid_mask].flatten(), minlength=19)
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        mean_iou = mean_iou_metric.compute()
        class_ious = class_iou_metric.compute()
        
        total_valid_pixels = pixel_counts.sum()
        pixel_percentages = (pixel_counts.float() / total_valid_pixels) * 100.0
        
        CITYSCAPES_CLASSES = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle'
        ]
        
        print("\n" + "="*40)
        print("--------- Evaluation Results ---------")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Mean IoU: {mean_iou.item():.4f}")
        print("--------- Per-Class IoU & Frequency ---------")
        
        results = {
            'loss': avg_loss,
            'mean_iou': mean_iou.item()
        }
        
        for i, iou in enumerate(class_ious):
            class_name = CITYSCAPES_CLASSES[i] if i < len(CITYSCAPES_CLASSES) else f"class_{i}"
            pct = pixel_percentages[i].item()
            results[f'iou_{class_name}'] = iou.item()
            results[f'pct_{class_name}'] = pct
            # Print with aligned formatting: ClassName: IoU (X.XX% of pixels)
            print(f"{class_name.capitalize():<15}: {iou.item():.4f}  |  Pixels: {pct:>5.2f}%")
            
        print("========================================\n")
        
        # Clean up
        mean_iou_metric.reset()
        class_iou_metric.reset()
        torch.cuda.empty_cache()
        
        return results

    def on_train_epoch_start(self) -> None:
        # Apply Gradual Unfreezing / Partial Fine-Tuning strategy.
        if self.hparams.unfreeze_epoch is not None and self.model.use_resnet and self.current_epoch == self.hparams.unfreeze_epoch:
            print("Unfreezing layers of ResNet34 (layer3 and bottleneck) for fine-tuning. Starting epoch: ", self.current_epoch)
            # Unfreeze the encoder layer3 and bottleneck (layer4) from the ResNet34.
            for params in self.model.encoder['layer3'].parameters():
                params.requires_grad = True
            for params in self.model.bottleneck.parameters():
                params.requires_grad = True 
    
    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        Frees the GPU memory cache to prevent memory fragmentation and memory creep over epochs.
        """
        torch.cuda.empty_cache()

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler for training.
        
        Returns
        -------
        dict
            A dictionary containing the optimizer and learning rate scheduler configuration.
        """
        # AdamW Optimizer
        enc_params = list(self.model.encoder.parameters()) + list(self.model.bottleneck.parameters())
        dec_params = list(self.model.decoder.parameters()) + list(self.model.output_conv.parameters())
        
        optimizer = optim.AdamW([{'params': enc_params, 'lr': self.hparams.enc_learning_rate}, 
                                 {'params': dec_params, 'lr': self.hparams.dec_learning_rate}],
                                 weight_decay=self.hparams.weight_decay, 
                                 foreach=False, fused=False)
        # Learning Rate Scheduler
        # 1. WarmUp lineal: Sube desde un 10% del LR inicial hasta el 100% durante las primeras 'warmup_epochs'
        warmup_epochs = self.hparams.warmup_epochs
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        
        # 2. Cosine Annealing LR
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(self.hparams.max_epochs - warmup_epochs), eta_min=1e-10
        )

        # 3. SequentialLR: Linear WarmUp + Cosine Annealing LR
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_epochs]
        )

        return{
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1
            },
        }

    def _overlap_tile(self, image: torch.Tensor, tile_size: tuple[int, int], stride: tuple[int, int]) -> torch.Tensor:
        """
        Creates the batch of overlapping tiles from an image.
        
        Parameters
        ----------
        image : torch.Tensor
            The input image of shape (1, C, H, W).
            Where the first dimension is the batch size (always 1).
        tile_size : tuple
            The size of the tiles to use for inference.
        stride : tuple
            The amount of stride between tiles.
        
        Returns
        -------
        torch.Tensor
            All the tiles grouped with shape tile_size
        """
        # Apply fold in dimension 2 (height) to get the tiles for each row and dimension 3 (width) to get the tiles for each column
        overlaping_tiles = image.unfold(dimension=2, size=tile_size[0], step=stride[0]).unfold(dimension=3, size=tile_size[1], step=stride[1])
        # Permute the dimensions from (B, C, num_tiles_h, num_tiles_w, tile_h, tile_w) to (B, num_tiles_h, num_tiles_w, C, tile_h, tile_w)
        # Reshape the dimension to get (B * num_tiles_h * num_tiles_w, C, tile_h, tile_w)
        overlaping_tiles = overlaping_tiles.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, image.shape[1], tile_size[0], tile_size[1])

        return overlaping_tiles
    
    
    def _reconstruct_tile(self, predictions: torch.Tensor, tile_size: tuple[int, int], 
                          stride: tuple[int, int], original_size: tuple[int, int]) -> torch.Tensor:
        """
        Reconstructs the full image from the batch of overlapping tiles.
        
        Parameters
        ----------
        predictions : torch.Tensor
            The predictions of the neural network of shape (B * num_tiles_h * num_tiles_w, C, tile_h, tile_w).
            Where the batch size is always 1.
        tile_size : tuple
            The size of the tiles to use for inference (HxW).
        stride : tuple
            The amount of stride between tiles (HxW).
        original_size : tuple
            The original size of the images (HxW).
        
        Returns
        -------
        torch.Tensor
            The neural network final prediction after reconstructing the image to its original size
        """
        num_tiles, num_classes, high, width = predictions.shape

        # Reshape the predictions to the correct shape for fold (1, C*H*W, num_tiles)
        # Input arrives in channels_last format; .contiguous() materializes to NCHW physical
        # order so that each tile's C*H*W elements are contiguous for the permute+view.
        predictions_nchw = predictions.contiguous()
        
        # Apply Gaussian weighting to the predictions of each tile
        predictions_weighted = predictions_nchw * self.gaussian_patch
        
        predictions_reshaped = predictions_weighted.permute(1, 2, 3, 0).reshape(1, num_classes*high*width, num_tiles)

        # Reconstruct the image from the tiles to the original size of the images in validation  
        reconstructed_tiles = torch.nn.functional.fold(input=predictions_reshaped,
                                                       output_size=original_size,
                                                       kernel_size=tile_size,
                                                       stride=stride)
        
        final_prediction = reconstructed_tiles / self.count_map
        return final_prediction
