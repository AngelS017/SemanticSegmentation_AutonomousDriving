import os
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import random
import shutil
from typing import Any
import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from CityscapesDataModule import CityscapesDataModule
from model.DriveSegmentationModel import U_Net
from model.DriveSegmentationLightningModule import DriveSegmentationLightningModule
from Dice_CrossEntropy_Loss import DiceCrossEntropyLoss

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import optuna.visualization as vis

from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import schedule

from torchinfo import summary


class OptunaMetricsCallback(Callback):
    """
    Custom callback to update Optuna user attributes at the end of each epoch.
    This allows monitoring metrics and the current epoch in real-time in the Optuna Dashboard.
    It also ensures pruned trials save their last known metrics.
    """
    def __init__(self, trial: optuna.Trial, checkpoint_callback: ModelCheckpoint):
        super().__init__()
        self.trial = trial
        self.checkpoint_callback = checkpoint_callback

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.trial is None:
            return
            
        metrics_to_report = ["train_loss", "train_iou", "val_loss", "val_iou"]
        for metric in metrics_to_report:
            value = trainer.callback_metrics.get(metric)
            if value is not None:
                self.trial.set_user_attr(metric, value.item())
        
        # Save current epoch to see where it stopped or is currently at
        self.trial.set_user_attr("current_epoch", trainer.current_epoch)
        
        # Save the path to the best checkpoint for this trial
        if self.checkpoint_callback.best_model_path:
            self.trial.set_user_attr("checkpoint_path", self.checkpoint_callback.best_model_path)


def set_improvement_techniques_training() -> None:
    """
    Sets some improvements for training the model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    pl.seed_everything(1234)

    # Set the internal precision for matrix multiplications to bfloat16 instead of float32
    torch.set_float32_matmul_precision('medium')
    # Enable the cuDNN auto-tuner to find the best algorithm for the convolution in the actual hardware
    torch.backends.cudnn.benchmark = True

    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.memory_planning = True

    torch._inductor.config.epilogue_fusion = True
    torch._inductor.config.conv_1x1_as_mm = True       
    torch.backends.cudnn.allow_tf32 = True               

def train_model(config: dict[str, Any], mean_dataset: list[float] | torch.Tensor, std_dataset: list[float] | torch.Tensor, trial: optuna.Trial = None) -> float:
    """
    Trains a PyTorch Lightning model using the provided datamodule.

    Optuna optimization passes a config dictionary containing the 
    hyperparameter combination of this trial for training.
    
    Parameters
    ----------
    config : dict
        A dictionary containing the hyperparameters for training. Must include:
        - "weight_dice" (float): Weight for the Dice component of the combined loss [0, 1].
        - "weight_classes" (list | torch.Tensor | None): Weights for each class for the loss component.
        - "enc_learning_rate" (float): Initial learning rate for the encoder.
        - "dec_learning_rate" (float): Initial learning rate for the decoder.
        - "weight_decay" (float): L2 regularization for AdamW.
        - "dropout_prob" (float): Dropout probability for the decoder blocks.
        - "freeze_encoder" (bool): Whether to freeze the ResNet34 encoder weights.
        - "batch_size" (int): Number of samples per training batch.
        - "crop_size" (tuple): Spatial size (H, W) of training crops.
        - "stride" (tuple): Stride (H, W) used during validation overlap-tiling.
        - "max_epochs" (int): Maximum number of training epochs.
        - "filters" (list | None): Channel sizes for custom U-Net encoder; None selects ResNet34.
        - "use_checkpointing" (bool | list[bool]): Gradient checkpointing per block.
        - "warmup_epochs" (int): Number of epochs for learning rate warmup.
        - "unfreeze_epoch" (int): Epoch at which to unfreeze the encoder.
        - "grad_accum" (int): Number of batches to accumulate gradients.
    mean_dataset : list or torch.Tensor
        The mean values of the dataset for normalization.
    std_dataset : list or torch.Tensor
        The standard deviation values of the dataset for normalization.
    trial : optuna.Trial, optional
        The Optuna trial object for pruning and tracking metrics, by default None.

    Returns
    -------
    float
        The best validation IoU (val_iou) achieved during training.
    """
    set_improvement_techniques_training()

    weight_classes = config["weight_classes"]
    if weight_classes is not None and not isinstance(weight_classes, torch.Tensor):
        weight_classes = torch.tensor(weight_classes, dtype=torch.float32)

    # Build the loss function
    loss = DiceCrossEntropyLoss(ignore_index=255, weight_dice=config["weight_dice"], 
                                weight_classes=weight_classes)
    
    model_backbone = U_Net(filters=config["filters"], dropout_prob=config["dropout_prob"], 
                           use_checkpointing=config["use_checkpointing"], 
                           freeze_encoder=config["freeze_encoder"])
    
    # Run abstract summary on CPU to avoid CUDA errors with checkpointing
    # ====================
    """
    model_summary = U_Net(filters=config["filters"], dropout_prob=config["dropout_prob"], 
                          use_checkpointing=False, 
                          freeze_encoder=config["freeze_encoder"])
    print(summary(model_summary, input_size=(config["batch_size"], 3, config["crop_size"][0], config["crop_size"][1]), 
            col_names=["input_size", "output_size", "num_params", "trainable", "kernel_size", "mult_adds", "params_percent"],
            row_settings=["var_names"], depth=10, device="cpu"))
    del model_summary
    """
    # ====================

    driveseg_model = DriveSegmentationLightningModule(model=model_backbone, 
                                                      loss=loss,
                                                      batch_size=config["batch_size"], 
                                                      enc_learning_rate=config["enc_learning_rate"],
                                                      dec_learning_rate=config["dec_learning_rate"],
                                                      max_epochs=config["max_epochs"],
                                                      weight_decay=config["weight_decay"],
                                                      crop_size=config["crop_size"],
                                                      stride=config["stride"],
                                                      warmup_epochs=config["warmup_epochs"],
                                                      unfreeze_epoch=config["unfreeze_epoch"])

    # Get the project root directory (one level up from utils)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir_absoluto = os.path.join(BASE_DIR, 'Cityscapes_data')

    datamodule = CityscapesDataModule(
        data_dir=data_dir_absoluto,
        batch_size=config["batch_size"],
        num_workers=15,
        mean=mean_dataset,
        std=std_dataset,
        crop_size=config["crop_size"]
    )

    # The trainer will handle the training, validation and testing loop.
    # And also the optimizer.zero_grad(), loss.backward() and optimizer.step() 
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator='gpu',
        devices=1,
        precision="bf16-mixed", # or 16-mixed
        accumulate_grad_batches=config["grad_accum"],
        deterministic=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        logger=True,
        gradient_clip_val=1.0
    )

    # Callbacks configuration
    callbacks = []
    if trial is not None:
        # Pruning callback
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_iou"))
        
        # Checkpoint callback to save the best model for this trial
        checkpoint_dir = os.path.join(BASE_DIR, "optuna_checkpoints_partially_frozen", f"trial_{trial.number}")
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model",
            monitor="val_iou",
            mode="max",
            save_top_k=1
        )
        callbacks.append(checkpoint_callback)
        
        # Custom callback to report metrics and epoch in real-time
        callbacks.append(OptunaMetricsCallback(trial, checkpoint_callback))
        
    trainer.callbacks.extend(callbacks)

    try:
        trainer.fit(driveseg_model, datamodule=datamodule)
    except optuna.exceptions.TrialPruned:
        # If the trial is pruned, delete its checkpoint folder to save disk space
        if trial is not None and 'checkpoint_dir' in locals() and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            # Update the user attribute to reflect that the file no longer exists
            trial.set_user_attr("checkpoint_path", "Deleted due to Pruning")
        raise  # Re-raise so Optuna registers the trial as pruned

    # Return the best metric for Optuna to optimize
    return trainer.callback_metrics.get("val_iou", torch.tensor(0.0)).item()



def set_hyperparameters_search_algorithm() -> optuna.samplers.BaseSampler:
    """
    Sets the hyperparameters search algorithm of Optuna.

    Returns
    -------
    optuna.samplers.BaseSampler
         The hyperparameters search algorithm to be used in Optuna.
    """
    return optuna.samplers.TPESampler(seed=1234)


def set_scheduler(min_resource: int = 10, reduction_factor: int = 2) -> optuna.pruners.BasePruner:
    """
    Sets the pruner for Optuna (ASHA equivalent).
    The pruner is used to stop trials that are not promising early on.

    Parameters
    ----------
    min_resource : int, optional
        The number of steps (epochs) to wait before considering stopping a trial (default is 10).
    reduction_factor : int, optional
        The factor by which to reduce the number of trials at each decision point (default is 2).
    
    Returns
    -------
    optuna.pruners.BasePruner
        The pruner (SuccessiveHalvingPruner) to be used in Optuna.
    """
    return optuna.pruners.SuccessiveHalvingPruner(min_resource=min_resource, reduction_factor=reduction_factor)


def objective(trial: optuna.Trial, hyperparameters_space: dict[str, Any], 
              mean_dataset: list[float] | torch.Tensor, std_dataset: list[float] | torch.Tensor) -> float:
    """
    Objective function for Optuna optimization.
    
    Parameters
    ----------
    trial : optuna.Trial
        The current Optuna trial.
    hyperparameters_space : dict
        The search space defining how to sample hyperparameters.
    mean_dataset : list or torch.Tensor
        Dataset mean for normalization.
    std_dataset : list or torch.Tensor
        Dataset standard deviation for normalization.
        
    Returns
    -------
    float
        The value of the metric to optimize (val_iou).
    """
    # Build config from space. If a value is a callable (lambda), call it with trial.
    config = {}
    for key, value in hyperparameters_space.items():
        if callable(value):
            config[key] = value(trial)
        else:
            config[key] = value
    
    return train_model(config, mean_dataset, std_dataset, trial=trial)


def train_tune_hyperparameters(hyperparameters_space: dict[str, Any],
                               mean_dataset: list[float] | torch.Tensor, std_dataset: list[float] | torch.Tensor, 
                               num_trials: int) -> optuna.Study:
    """
    Runs hyperparameter search with Optuna.

    Parameters
    ----------
    hyperparameters_space : dict
        A dictionary defining the hyperparameters search space.
        Values can be fixed or functions (lambdas) that take a trial and return a value.
    mean_dataset : list or torch.Tensor
        The mean values of the dataset for normalization.
    std_dataset : list or torch.Tensor
        The standard deviation values of the dataset for normalization.
    num_trials : int
        The number of hyperparameter trials to run.

    Returns
    -------
    optuna.Study
        The Optuna study object.
    """
    # Use SQLite for persistence and to allow optuna-dashboard to connect
    storage_name = "sqlite:///optuna_search.db"
    study_name = "semantic_segmentation_tuning_partially_frozen"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        sampler=set_hyperparameters_search_algorithm(),
        pruner=set_scheduler(min_resource=15),
        load_if_exists=True
    )

    print(f"Starting Optuna optimization. You can monitor progress with:")
    print(f"  optuna-dashboard {storage_name}")

    # Call the objective function using a lambda to pass extra arguments
    study.optimize(lambda trial: objective(trial, hyperparameters_space, mean_dataset, std_dataset), 
                   n_trials=num_trials)

    # Save visualization plots
    try:
        vis.plot_optimization_history(study).write_html("optuna_history.html")
        vis.plot_param_importances(study).write_html("optuna_importances.html")
        vis.plot_parallel_coordinate(study).write_html("optuna_parallel.html")
        print("Optimization plots saved as HTML files.")
    except Exception as e:
        print(f"Could not save plots: {e}")

    return study


def model_profiler(hyperparameters_space: dict[str, Any], loss: torch.nn.Module, 
                   mean_dataset: list[float] | torch.Tensor, std_dataset: list[float] | torch.Tensor, 
                   class_weights: torch.Tensor | None = None) -> PyTorchProfiler:
    """
    Function to profile the model.

    Parameters
    ----------
    hyperparameters_space : dict
        A dictionary defining the hyperparameters of the model. Expects keys similar to `config` in `train_model`.
    loss : torch.nn.Module
        The loss function to be used in the training loop.
    mean_dataset : list or torch.Tensor
        The mean values of the dataset for normalization.
    std_dataset : list or torch.Tensor
        The standard deviation values of the dataset for normalization.
    class_weights : torch.Tensor | None, optional
        Weights for each class to be used in the loss function, by default None. (Note: currently unused in the function body).

    Returns
    -------
    PyTorchProfiler
        The profiler report containing the results of the model profiling.
    """
    set_improvement_techniques_training()

    # Get the project root directory (one level up from utils)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profiler_dir = os.path.join(BASE_DIR, 'lightning_profiler_logs')
    os.makedirs(profiler_dir, exist_ok=True)
    
    data_dir_absoluto = os.path.join(BASE_DIR, 'Cityscapes_data')
    # Set model
    model_backbone = U_Net(filters=hyperparameters_space["filters"], dropout_prob=hyperparameters_space["dropout_prob"], 
                           use_checkpointing=hyperparameters_space["use_checkpointing"])
    driveseg_model = DriveSegmentationLightningModule(model=model_backbone, 
                                                      loss=loss,
                                                      batch_size=hyperparameters_space["batch_size"],
                                                      enc_learning_rate=hyperparameters_space["enc_learning_rate"],
                                                      dec_learning_rate=hyperparameters_space["dec_learning_rate"],
                                                      max_epochs=1,
                                                      weight_decay=hyperparameters_space["weight_decay"],
                                                      crop_size=hyperparameters_space["crop_size"],
                                                      stride=hyperparameters_space["stride"],
                                                      warmup_epochs=hyperparameters_space["warmup_epochs"],
                                                      unfreeze_epoch=hyperparameters_space["unfreeze_epoch"])
    # Set datamodule
    datamodule = CityscapesDataModule(
        data_dir=data_dir_absoluto,
        batch_size=hyperparameters_space["batch_size"],
        num_workers=8,
        mean=mean_dataset,
        std=std_dataset,
        crop_size=hyperparameters_space["crop_size"]
    )
    # Set Profiler
    profiler = PyTorchProfiler(
        # Set the directory to save the profiler report
        dirpath=profiler_dir,
        # Specify the filename for the report
        filename="profile_report",
        # Define the profiling schedule (wait -> warmup -> active)
        # Total 25 steps
        schedule=schedule(wait=10, warmup=10, active=40, repeat=1),
        # Enable memory usage profiling
        profile_memory=True,
        with_stack=True,
    )
    # Set and train the model with the profiler
    trainer = pl.Trainer(
        profiler=profiler,
        max_steps=60,
        accelerator='gpu',
        devices=1,
        precision="bf16-mixed", # 16-mixed
        accumulate_grad_batches=hyperparameters_space["grad_accum"],
        deterministic=False, # For reproducibility
        enable_model_summary=True,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(driveseg_model, datamodule=datamodule)

    return profiler