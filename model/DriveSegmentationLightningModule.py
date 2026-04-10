import lightning.pytorch as pl
import torch
from torch import optim
from torchmetrics.classification import MulticlassJaccardIndex

class DriveSegmentationLightningModule(pl.LightningModule):
    def __init__(self, model, loss, learning_rate, max_epochs, weight_decay=1e-2):
        super().__init__()

        # Save the hyperparameters passed to the constructor, access via `self.hparams`
        # We don't want to save the model and loss function as hyperparameters
        self.save_hyperparameters(ignore=['model', 'loss'])

        model = model.to(memory_format=torch.channels_last)  
        self.model = torch.compile(model, dynamic=False)
        self.loss = loss

        # Set other metrics to track
        # The mean IoU metric is calculated as the average of the IoU for each class. 
        self.mean_iou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255, average='macro')
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        """
        Defines a single step of training.
        
        Parameters
        ----------
        batch : tuple
            A batch of data from DataLoader, typically a tuple of (inputs, targets).
        batch_idx : int
            The index of the current batch. Lightning Trainer requires this argument 
        
        """
        images, targets = batch
        images = images.to(memory_format=torch.channels_last, non_blocking=True)

        outputs_logits = self.forward(images)
        
        # Calculate the loss
        loss = self.loss(outputs_logits, targets)
        # Log the training loss and iou metric at the end of each epoch
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx=None):
        """
        Defines a single step of validation.  
        
        Parameters        
        ----------
        batch : tuple
            A batch of data from DataLoader, typically a tuple of (inputs, targets).
        batch_idx : int
            The index of the current batch. Lightning Trainer requires this argument 
        """
        images, targets = batch
        images = images.to(memory_format=torch.channels_last, non_blocking=True)


        outputs_logits = self.forward(images)

        class_prediction = torch.argmax(outputs_logits, dim=1)
        
        # Calculate the loss
        loss = self.loss(outputs_logits, targets)
        # Calulate the IoU metric
        self.mean_iou_metric(class_prediction, targets)
        # Log the val loss and iou metric at the end of each epoch
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_iou', self.mean_iou_metric, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx=None):
        """
        Defines a single step of testing.
        
        Parameters
        ----------
        batch : tuple
            A batch of data from DataLoader, typically a tuple of (inputs, targets).
        batch_idx : int
            The index of the current batch. Lightning Trainer requires this argument
        """
        images, targets = batch
        images = images.to(memory_format=torch.channels_last)

        outputs_logits = self.forward(images)

        class_prediction = torch.argmax(outputs_logits, dim=1)
        
        # Calculate the loss
        loss = self.loss(outputs_logits, targets)
        # Calulate the IoU metric
        self.mean_iou_metric(class_prediction, targets)
        # Log the test loss and iou metric at the end of each epoch
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_iou', self.mean_iou_metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for training.
        
        Returns
        -------
        dict
            A dictionary containing the optimizer and learning rate scheduler configuration.
        """
        # Optimizer algorithm
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, fused=True)
        # Learning Rate Scheduler
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=0.0) 

        return{
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1
            },
        }

    """
    def on_train_epoch_end(self):
        # Empty the GPU cache at the end of each training epoch to free up memory
        torch.cuda.empty_cache()
    """