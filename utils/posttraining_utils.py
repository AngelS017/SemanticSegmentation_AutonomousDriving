import numpy as np
from tqdm import tqdm
import os
import torch
import shutil
import lightning.pytorch as pl
import onnxruntime as ort
import cv2
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from model.DriveSegmentationModel import U_Net
from model.DriveSegmentationLightningModule import DriveSegmentationLightningModule

CITYSCAPES_PALETTE = np.array([
        [128,  64, 128],   # 0  road
        [244,  35, 232],   # 1  sidewalk
        [ 70,  70,  70],   # 2  building
        [102, 102, 156],   # 3  wall
        [190, 153, 153],   # 4  fence
        [153, 153, 153],   # 5  pole
        [250, 170,  30],   # 6  traffic light
        [220, 220,   0],   # 7  traffic sign
        [107, 142,  35],   # 8  vegetation
        [152, 251, 152],   # 9  terrain
        [ 70, 130, 180],   # 10 sky
        [220,  20,  60],   # 11 person
        [255,   0,   0],   # 12 rider
        [  0,   0, 142],   # 13 car
        [  0,   0,  70],   # 14 truck
        [  0,  60, 100],   # 15 bus
        [  0,  80, 100],   # 16 train
        [  0,   0, 230],   # 17 motorcycle
        [119,  11,  32],   # 18 bicycle
    ], dtype=np.uint8)


def save_checkpoint(checkpoint_base_path, save_path):
    """
    Saves or persists a checkpoint to a custom path.

    Parameters
    ----------
    checkpoint_base_path : str
        Path to the best checkpoint from Optuna.
    save_path : str
        The destination path where the checkpoint will be stored.

    Returns
    -------
    None
    """
    # Create intermediate directories if they don't exist
    target_path = os.path.join(checkpoint_base_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.isfile(target_path):
        shutil.copy(target_path, save_path)
        print(f"File copied! Renamed from 'checkpoint' to 'best_model.ckpt'")


def load_checkpoint(checkpoint_path, model, loss, device="cuda"):
    """
    Loads a Lightning checkpoint from disk and returns the LightningModule
    ready for inference with trainer.validate() or trainer.test().

    Since 'model' and 'loss' were excluded from save_hyperparameters() during
    training, they must be provided when loading the checkpoint.
    The remaining hyperparameters (enc_learning_rate, dec_learning_rate, crop_size, stride, etc.) are
    automatically restored from the checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the previously saved .ckpt file.
    model : torch.nn.Module
        An instance of the model (U_Net) with the same architecture used
        during training.
    loss : torch.nn.Module
        The loss function used during training.
    device : str, optional
        Device to load the model onto ('cpu' or 'cuda'), defaults to 'cuda'.

    Returns
    -------
    DriveSegmentationLightningModule
        The Lightning module with loaded weights, in evaluation mode
        and ready for use with trainer.validate() or trainer.test().

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist at the specified path.

    Examples
    --------
    >>> from model.DriveSegmentationModel import U_Net
    >>> import torch.nn as nn
    >>>
    >>> model = U_Net(filters=[64, 128, 256, 512, 1024], dropout_prob=0.0)
    >>> loss = nn.CrossEntropyLoss(ignore_index=255)
    >>> lightning_module = load_checkpoint(
    ...     checkpoint_path='/home/user/models/unet_best.ckpt',
    ...     model=model,
    ...     loss=loss,
    ...     device='cuda'
    ... )
    >>>
    >>> trainer = pl.Trainer(...)
    >>> trainer.validate(lightning_module, datamodule=datamodule)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}"
        )

    lightning_module = DriveSegmentationLightningModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        loss=loss,
        map_location=device
    )

    lightning_module.eval()
    print(f"Checkpoint loaded from: {checkpoint_path} (device: {device})")

    return lightning_module

def predict_pytorch_example(model, mean, std, device='cuda'):
    # Images to predict
    image_frankfurt_path = './Cityscapes_data/leftImg8bit/val/frankfurt/frankfurt_000000_015676_leftImg8bit.png'
    image_lindau_path = './Cityscapes_data/leftImg8bit/val/lindau/lindau_000024_000019_leftImg8bit.png'
    image_munster_path = './Cityscapes_data/leftImg8bit/val/munster/munster_000166_000019_leftImg8bit.png'

    # Ground truth masks
    mask_frankfurt_path = './Cityscapes_data/gtFine/val/frankfurt/frankfurt_000000_015676_gtFine_color.png'
    mask_lindau_path = './Cityscapes_data/gtFine/val/lindau/lindau_000024_000019_gtFine_color.png'
    mask_munster_path = './Cityscapes_data/gtFine/val/munster/munster_000166_000019_gtFine_color.png'
    
    # Set model to eval and preprocessing transforms
    model.eval().to(device)
    val_test_transform_compose = A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ])

    # Read images
    image_frankfurt_original = cv2.imread(image_frankfurt_path)
    image_lindau_original = cv2.imread(image_lindau_path)
    image_munster_original = cv2.imread(image_munster_path)
    # Apply transforms to images
    image_frankfurt = val_test_transform_compose(image=image_frankfurt_original)['image']
    image_lindau = val_test_transform_compose(image=image_lindau_original)['image']
    image_munster = val_test_transform_compose(image=image_munster_original)['image']

    # Apply batch dimension an set images in the device
    image_frankfurt = image_frankfurt.unsqueeze(0).to(device)
    image_lindau = image_lindau.unsqueeze(0).to(device)
    image_munster = image_munster.unsqueeze(0).to(device)

    with torch.no_grad():
        output_frankfurt = model(image_frankfurt)
        output_lindau = model(image_lindau)
        output_munster = model(image_munster)

        # Get predicted class for each image
        pred_class_frankfurt = torch.argmax(output_frankfurt, dim=1).squeeze(0)
        pred_class_lindau = torch.argmax(output_lindau, dim=1).squeeze(0)
        pred_class_munster = torch.argmax(output_munster, dim=1).squeeze(0)

    # Apply color map to the predicted classes
    pred_class_frankfurt_color = CITYSCAPES_PALETTE[pred_class_frankfurt.cpu().numpy()]
    pred_class_lindau_color = CITYSCAPES_PALETTE[pred_class_lindau.cpu().numpy()]
    pred_class_munster_color = CITYSCAPES_PALETTE[pred_class_munster.cpu().numpy()]

    # Read ground truth masks and convert to RGB
    mask_frankfurt_color = cv2.cvtColor(cv2.imread(mask_frankfurt_path), cv2.COLOR_BGR2RGB)
    mask_lindau_color = cv2.cvtColor(cv2.imread(mask_lindau_path), cv2.COLOR_BGR2RGB)
    mask_munster_color = cv2.cvtColor(cv2.imread(mask_munster_path), cv2.COLOR_BGR2RGB)

    # Convert original images to RGB for plotting (to avoid bluish tint from BGR)
    image_frankfurt_rgb = cv2.cvtColor(image_frankfurt_original, cv2.COLOR_BGR2RGB)
    image_lindau_rgb = cv2.cvtColor(image_lindau_original, cv2.COLOR_BGR2RGB)
    image_munster_rgb = cv2.cvtColor(image_munster_original, cv2.COLOR_BGR2RGB)

    # Plot the original images, ground truth masks, and the predicted classes
    fig, axes = plt.subplots(3, 3, figsize=(26, 16))
    
    # Frankfurt
    axes[0, 0].imshow(image_frankfurt_rgb)
    axes[0, 0].set_title("Frankfurt")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_frankfurt_color)
    axes[0, 1].set_title("Ground Truth Frankfurt")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(pred_class_frankfurt_color)
    axes[0, 2].set_title("Predicted Class Frankfurt")
    axes[0, 2].axis("off")
    
    # Lindau
    axes[1, 0].imshow(image_lindau_rgb)
    axes[1, 0].set_title("Lindau")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(mask_lindau_color)
    axes[1, 1].set_title("Ground Truth Lindau")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(pred_class_lindau_color)
    axes[1, 2].set_title("Predicted Class Lindau")
    axes[1, 2].axis("off")
    
    # Munster
    axes[2, 0].imshow(image_munster_rgb)
    axes[2, 0].set_title("Munster")
    axes[2, 0].axis("off")
    axes[2, 1].imshow(mask_munster_color)
    axes[2, 1].set_title("Ground Truth Munster")
    axes[2, 1].axis("off")
    axes[2, 2].imshow(pred_class_munster_color)
    axes[2, 2].set_title("Predicted Class Munster")
    axes[2, 2].axis("off")
    
    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    plt.show()


    
    

def export_model_onnx(model, output_path, name_model, input_size=(1, 3, 1024, 2048)):
    """
    Exports a PyTorch model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to export.
    output_path : str
        The path where the ONNX model will be saved.
    name_model : str
        The name of the ONNX model.
    input_size : tuple, optional
        The input size of the model (batch_size, channels, height, width),
        defaults to (1, 3, 1024, 2048).

    Returns
    -------
    None
    """
    os.makedirs(output_path, exist_ok=True)
    export_file = os.path.join(output_path, name_model)
    
    # Unpack model if it was wrapped by torch.compile()
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Get the precision and device of the model for the dummy input
    first_param = next(model.parameters())
    device = first_param.device
    dtype = first_param.dtype
    
    dummy_input = torch.randint(0, 256, input_size, dtype=torch.uint8, device=device)
    
    print("Exporting model to ONNX...")
    print(f"  Path: {export_file}")
    print(f"  Input shape: {input_size}")
    print(f"  Device: {device}")
    print(f"  Model parameters precision: {dtype}")
    print(f"  Input data type: {dummy_input.dtype}")
    
    # Export to onnx
    torch.onnx.export(model, 
                      (dummy_input,),
                      export_file,
                      optimize=True,
                      export_params=True,
                      opset_version=21,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 
                                    'output': {0: 'batch_size'}})

def predict_onnx(model_path, images_dir, output_dir, mean, std):
    # 1. Convert mean/std to numpy arrays for the preprocessing pipeline
    if isinstance(mean, torch.Tensor):
        mean = mean.numpy()
    if isinstance(std, torch.Tensor):
        std = std.numpy()
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    # 2. Discover all images in the input directory (recursive)
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = []
    for dirpath, _, filenames in os.walk(images_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(dirpath, filename))
    image_paths.sort()

    if len(image_paths) == 0:
        print(f"No images found in: {images_dir}")
        return

    print(f"Found {len(image_paths)} images in '{images_dir}'")

    # 3. Create ONNX Runtime session with TensorRT acceleration
    try:
        so = ort.SessionOptions()
        # Disable ONNX graph-level optimizations so TensorRT handles them
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(
            model_path, so,
            providers=[
                ('TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.path.dirname(model_path),
                    'trt_fp16_enable': True,
                    'trt_cuda_graph_enable': False,
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        )
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Verify which provider was selected
    active_providers = session.get_providers()
    print(f"Active ONNX Runtime providers: {active_providers}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    os.makedirs(output_dir, exist_ok=True)

    # 4. Inference loop — image by image (simulating a camera stream)
    io = session.io_binding()
    inference_times = []

    for img_path in tqdm(image_paths, desc="ONNX Inference"):
        # Read the image (BGR)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [WARNING] Could not read: {img_path}, skipping.")
            continue

        # Keep a copy for the overlay visualisation
        img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb_original, (2, 0, 1)) # HWC → CHW
        img_batch = np.expand_dims(img_chw, axis=0)         # (1,C,H,W)

        # Upload to GPU via OrtValue and run inference
        img_ort = ort.OrtValue.ortvalue_from_numpy(
            img_batch, device_type="cuda", device_id=0
        )

        io.bind_input(
            name=input_name,
            device_type="cuda", device_id=0,
            element_type=np.uint8,
            shape=img_batch.shape,
            buffer_ptr=img_ort.data_ptr(),
        )
        io.bind_output(name=output_name, device_type="cuda", device_id=0)

        t_start = time.perf_counter()
        # Run inference
        session.run_with_iobinding(io)
        t_end = time.perf_counter()
        inference_times.append(t_end - t_start)
        # Retrieve the prediction (class indices) — copy from GPU to CPU
        pred = io.copy_outputs_to_cpu()[0]              # (1, H, W) uint8
        pred_map = pred.squeeze(0)                      # (H, W)
        # Colorise the segmentation mask
        colored_mask = CITYSCAPES_PALETTE[pred_map]     # (H, W, 3) RGB
        # Create an overlay: original + mask blended
        alpha = 0.5
        overlay = cv2.addWeighted(
            img_rgb_original, 1 - alpha,
            colored_mask, alpha,
            0,
        )  
        # Save results
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Preserve directory structure
        rel_path = os.path.relpath(os.path.dirname(img_path), images_dir)
        current_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)
        
        overlay_path = os.path.join(current_output_dir, f"{base_name}_overlay.png")
        pred_path = os.path.join(current_output_dir, f"{base_name}_pred.png")
        # Save as BGR for OpenCV
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        # Guardar la máscara coloreada en lugar de los índices (que se ven negros)
        cv2.imwrite(pred_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        # Clear bindings for the next iteration
        io.clear_binding_inputs()
        io.clear_binding_outputs()
    
    # 5. Summary statistics
    if inference_times:
        times = np.array(inference_times)
        print(f"\n{'='*50}")
        print(f"  Inference complete — {len(inference_times)} images")
        print(f"  Results saved to: {output_dir}")
        print(f"{'='*50}")
        print(f"  Avg time per frame : {times.mean()*1000:.2f} ms")
        print(f"  Median             : {np.median(times)*1000:.2f} ms")
        print(f"  Min / Max          : {times.min()*1000:.2f} / {times.max()*1000:.2f} ms")
        print(f"  Throughput (avg)   : {1.0/times.mean():.1f} FPS")
        print(f"{'='*50}")