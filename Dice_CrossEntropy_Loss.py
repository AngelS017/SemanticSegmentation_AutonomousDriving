import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    """
    Computes the Dice Loss for multiclass semantic segmentation.
    
    The Dice Loss is based on the overlap between the prediction and the ground truth,
    and is particularly useful for handling class imbalance.

    The formula for the Dice Coefficient for each class is:
        Dice = (2 * Σ(y_pred * y_true) + smooth) / (Σy_pred + Σy_true + smooth)
    
    The Dice Loss is then calculated as:
        DiceLoss = 1 - Mean(Dice)
    """
    def __init__(self, smooth: float =1e-5, ignore_index: int =255):
        """
        Initializes the MulticlassDiceLoss module.

        Parameters
        ----------
        smooth : float, optional
            A small constant added to the numerator and denominator to prevent division 
            by zero (default is 1e-5).
        ignore_index : int, optional
            The label index to ignore during loss calculation (e.g., 255 for unlabeled 
            pixels) (default is 255).
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, probs: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the multiclass Dice Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted logits from the model, with shape (N, C, H, W).
        y_true : torch.Tensor
            The ground truth labels, with shape (N, H, W).
        probs : torch.Tensor, optional
            Pre-computed probabilities (Softmax). If provided, y_pred is ignored.
            Useful for avoiding redundant calculations in combined losses.

        Returns
        -------
        torch.Tensor
            The calculated mean Dice Loss across all valid classes.
        """
        # If probs are not pre-computed, calculate softmax
        if probs is None:
            probs = F.softmax(y_pred, dim=1)
            
        num_classes = probs.shape[1]
        
        # Mask valid pixels: (batch_size, 1, height, width)
        valid_mask = (y_true != self.ignore_index).unsqueeze(1)
        
        # Optimization: Avoid clone(), use masked_fill for safe one-hot indexing
        y_true_safe = y_true.masked_fill(y_true == self.ignore_index, 0)
        
        # One-hot: (batch_size, num_classes, height, width)
        target_one_hot = F.one_hot(y_true_safe, num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=probs.dtype)
        
        # Apply mask once to reduce memory bandwidth usage
        probs_masked = probs * valid_mask
        target_masked = target_one_hot * valid_mask

        # Calculate intersection and union
        intersection = torch.sum(probs_masked * target_masked, dim=(0, 2, 3))
        union = torch.sum(probs_masked, dim=(0, 2, 3)) + torch.sum(target_masked, dim=(0, 2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - torch.mean(dice_score)


class DiceCrossEntropyLoss(nn.Module):
    """
    A combined loss function that computes a weighted sum of Dice Loss and Cross-Entropy Loss.
    
    This combination helps leverage the strengths of both losses: Dice Loss for handling 
    class imbalance and Cross-Entropy Loss for pixel-wise classification accuracy.
    """
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255, weight_dice: float = 0.5, weight_classes: torch.Tensor = None):
        """
        Initializes the DiceCrossEntropyLoss module.

        Parameters
        ----------
        smooth : float, optional
            A small constant for smoothing the Dice Loss calculation (default is 1e-5).
        ignore_index : int, optional
            The label index to ignore during loss calculation (default is 255).
        weight_dice : float, optional
            The weight assigned to the Dice Loss component. The Cross-Entropy Loss will 
            be weighted as (1 - weight_dice) (default is 0.5).
        weight_classes : torch.Tensor, optional
            A manual rescaling weight given to each class in the Cross-Entropy loss.
            If given, has to be a Tensor of size `C` (number of classes).
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight_dice = weight_dice
        self.weight_ce = 1 - weight_dice
        
        # Register weight_classes as a buffer so it automatically moves to the correct device (GPU/CPU)
        self.register_buffer('weight_classes', weight_classes)

        # Optimized sub-module
        self.dice_loss_fn = MulticlassDiceLoss(smooth=smooth, ignore_index=ignore_index)

        # Disabling compilation for better error reporting and to isolate the device-side assert issues
        # self.forward = torch.compile(self.forward)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined Dice and Cross-Entropy Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted logits from the model, with shape (N, C, H, W).
        y_true : torch.Tensor
            The ground truth labels, with shape (N, H, W).

        Returns
        -------
        torch.Tensor
            The weighted combination of Dice Loss and Cross-Entropy Loss.
        """
        # LogSoftmax is necessary for NLL (CrossEntropy) and its exp is more 
        # numerically stable than direct Softmax for the Dice component.
        log_probs = F.log_softmax(y_pred, dim=1)
        
        # 1. Calculate Cross-Entropy using the shared log_probs
        ce_loss = F.nll_loss(log_probs, y_true, ignore_index=self.ignore_index, weight=self.weight_classes)
        
        # 2. Calculate Dice Loss using probabilities derived from shared log_probs
        probs = torch.exp(log_probs)
        dice_loss = self.dice_loss_fn(y_pred, y_true, probs=probs)
        
        # 3. Weighted sum
        return (self.weight_dice * dice_loss) + (self.weight_ce * ce_loss)    
        
        

        

        