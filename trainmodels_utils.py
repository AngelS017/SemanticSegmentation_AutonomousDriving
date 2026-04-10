import numpy as np
import torch
import os
import lightning.pytorch as pl

from CityscapesDataModule import CityscapesDataModule
from model.DriveSegmentationModel import U_Net
from model.DriveSegmentationLightningModule import DriveSegmentationLightningModule

from ray import tune
# Callback of Pytorch Lighning to report metrics to Ray Tune during training
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import schedule

from torchinfo import summary

def set_improvement_techniques_training():
    """
    Sets some improvements for training the model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    np.random.seed(1234)
    torch.manual_seed(1234)
    pl.seed_everything(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True 

    # Set the internal precision for matrix multiplications to bfloat16 instead of float32
    torch.set_float32_matmul_precision('medium')
    # Enable the cuDNN auto-tuner to find the best algorithm for the convolution in the actual hardware
    torch.backends.cudnn.benchmark = True

    #torch._inductor.config.fx_graph_cache = True 
    #torch._inductor.config.memory_planning = False


def train_model(config, loss, mean_dataset, std_dataset):
    """
    Trains a PyTorch Lightning model using the provided datamodule.

    Ray Tune stablish the first argument of the train_model function as the config, which is a dictionary containing the 
    hyperparameters combination of this trial for training.
    
    Parameters
    ----------
    config : dict
        A dictionary containing the hyperparameters for training.
    loss : torch.nn.Module
        The loss function to be used in the training loop.
    mean_dataset : list
        The mean values of the dataset for normalization.
    std_dataset : list
        The standard deviation values of the dataset for normalization.

    Returns
    -------
    None
    """
    set_improvement_techniques_training()
    
    model_backbone = U_Net(filters=config["filters"], dropout_prob=config["dropout_prob"], 
                           use_checkpointing=config["use_checkpointing"])
    
    # Run abstract summary on CPU to avoid CUDA errors with checkpointing
    # ====================
    model_summary = U_Net(filters=config["filters"], dropout_prob=config["dropout_prob"], 
                          use_checkpointing=False)
    summary(model_summary, input_size=(config["batch_size"], 3, config["crop_size"][0], config["crop_size"][1]), 
            col_names=["input_size", "output_size", "num_params", "trainable", "kernel_size", "mult_adds", "params_percent"],
            row_settings=["var_names"], depth=10, device="cpu")
    del model_summary
    # ====================

    driveseg_model = DriveSegmentationLightningModule(model=model_backbone, 
                                                      loss=loss, 
                                                      learning_rate=config["learning_rate"],
                                                      max_epochs=config["max_epochs"],
                                                      weight_decay=config["weight_decay"])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir_absoluto = os.path.join(BASE_DIR, 'Cityscapes_data')

    datamodule = CityscapesDataModule(
        data_dir=data_dir_absoluto,
        batch_size=config["batch_size"],
        num_workers=8,
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
        precision="bf16-mixed", # 16-mixed
        accumulate_grad_batches=config["grad_accum"],
        deterministic=False, # For reproducibility
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=True,
        callbacks=[TuneReportCheckpointCallback()]
    )

    trainer.fit(driveseg_model, datamodule=datamodule)


def set_hyperparameters_search_algorithm():
    """
    Sets the hyperparameters search algorithm of Ray Tune.

    Returns
    -------
    OptunaSearch
         The hyperparameters search algorithm to be used in Ray Tune.
    """
    return OptunaSearch()


def set_scheduler(max_t=80, grace_period=10, reduction_factor=2):
    """
    Sets the scheduler for the hyperparameters search algorithm of Ray Tune.
    The scheduler is used to control the execution of each trial in the hyperparameters search been able to
    stop trials that are not promising early on.

    Each decison point is determined by the grace_period and the reduction_factor. Where the first one is only the value of 
    grace_period, the others can be calculated multiplying the previous value by the reduction_factor.
    Decision points with default values: epoch 10, 20, 40, 80.

    Set the max_t = max_epochs of the training loop.

    Parameters
    ----------
    max_t : int, optional
        The maximum number of epochs to run for each trial (default is 80).
    grace_period : int, optional
        The number of epochs to wait before considering stopping a trial (default is 10).
    reduction_factor : int, optional
        The factor by which to reduce the number of trials at each decision point (default is 2).
    
    Returns
    -------
    ASHAScheduler
        The scheduler to be used in Ray Tune.
    """
    return ASHAScheduler(time_attr="epoch", max_t=max_t, grace_period=grace_period, 
                         reduction_factor=reduction_factor)


def train_tune_hyperparameters(hyperparameters_space, loss, mean_dataset, std_dataset, num_trials, gpu_per_trial=1.0, cpu_per_trial=8):
    """
    Placeholder function for training with hyperparameters tuning using Ray Tune.

    Parameters
    ----------
    hyperparameters_space : dict
        A dictionary defining the hyperparameters search space for Ray Tune.
    loss : torch.nn.Module
        The loss function to be used in the training loop.
    mean_dataset : list
        The mean values of the dataset for normalization.
    std_dataset : list
        The standard deviation values of the dataset for normalization.
    num_trials : int
        The number of hyperparameters trials to run in Ray Tune.

    Returns
    -------
    tune.ResultGrid
        The result grid containing the results of the hyperparameters tuning.
    """
    trainable_train_model = tune.with_parameters(train_model, 
                                                 loss=loss, 
                                                 mean_dataset=mean_dataset, 
                                                 std_dataset=std_dataset)
    train_model_with_resources = tune.with_resources(trainable=trainable_train_model, 
                                                     resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial})
    
    scheduler = set_scheduler(max_t=hyperparameters_space["max_epochs"])
    search_algorithm = set_hyperparameters_search_algorithm()

    tuner = tune.Tuner(
        trainable=train_model_with_resources,
        param_space=hyperparameters_space,
        tune_config=tune.TuneConfig(metric="val_loss", 
                                    mode="min",
                                    scheduler=scheduler, 
                                    search_alg=search_algorithm, 
                                    num_samples=num_trials),
        run_config=tune.RunConfig(verbose=1)
    )

    return tuner.fit()


def model_profiler(hyperparameters_space, loss, mean_dataset, std_dataset, class_weights=None):
    """
    Function to profile the model.

    Parameters
    ----------
    hyperparameters_space : dict
        A dictionary defining the hyperparameters of the model.
    loss : torch.nn.Module
        The loss function to be used in the training loop.
    mean_dataset : list
        The mean values of the dataset for normalization.
    std_dataset : list
        The standard deviation values of the dataset for normalization.

    Returns
    -------
    profiler_report : PyTorchProfiler
        The profiler report containing the results of the model profiling.
    """
    set_improvement_techniques_training()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    profiler_dir = os.path.join(BASE_DIR, 'lightning_profiler_logs')
    os.makedirs(profiler_dir, exist_ok=True)
    
    data_dir_absoluto = os.path.join(BASE_DIR, 'Cityscapes_data')
    # Set model
    model_backbone = U_Net(filters=hyperparameters_space["filters"], dropout_prob=hyperparameters_space["dropout_prob"], 
                           use_checkpointing=hyperparameters_space["use_checkpointing"])
    driveseg_model = DriveSegmentationLightningModule(model=model_backbone, 
                                                      loss=loss, 
                                                      learning_rate=hyperparameters_space["learning_rate"],
                                                      max_epochs=1,
                                                      weight_decay=hyperparameters_space["weight_decay"])
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