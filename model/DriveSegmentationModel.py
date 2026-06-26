import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet34, ResNet34_Weights


class DownsamplingBlock(nn.Module):
    """
    Downsampling block for the encoder part of the U-Net architecture.
    It consists of a convolutional block (Convolutions + ReLU + Dropout) 
    followed by a max pooling layer.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_prob : float, optional
        Dropout probability, by default 0.2.
    max_pooling : bool, optional
        Whether to apply max pooling, by default True.
    use_checkpointing : bool, optional
        Whether to use gradient checkpointing, by default False.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.2, max_pooling: bool = True, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        # Main convolutional block (Convolutions + ReLU + Dropout)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Weight initialization he_normal (Kaiming Normal) for Conv layers
        for model_layer in self.conv_block.modules():
            if isinstance(model_layer, nn.Conv2d):
                nn.init.kaiming_normal_(model_layer.weight, nonlinearity='relu')
                if model_layer.bias is not None:
                    nn.init.constant_(model_layer.bias, 0)

        # Max pooling layer (applied after skip connection)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if max_pooling else nn.Identity()

        # Dropout layer (applied after skip connection)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        # 1. Convolutions + ReLU + Dropout (this output is our skip connection)
        if self.use_checkpointing and self.training:
            skip_connection = checkpoint(self.conv_block, x, use_reentrant=False, preserve_rng_state=False)
        else:
            skip_connection = self.conv_block(x)
        
        # 2. Max Pooling (run in eager mode to avoid torchinductor bug with MaxPool2d backward + bf16)
        next_layer = self._eager_maxpool(skip_connection)

        # 3. Dropout
        next_layer = self.dropout(next_layer)

        return next_layer, skip_connection

    @torch.compiler.disable
    def _eager_maxpool(self, x):
        """Executes MaxPool2d outside of torch.compile to prevent a torchinductor bug
        where the fused Triton kernel for max_pool2d_with_indices backward generates
        out-of-bounds indices (assertion: 0 <= index < 4) with bf16 mixed precision."""
        return self.maxpool(x)


class UpsamplingBlock(nn.Module):
    """
    Upsampling block for the decoder part of the U-Net architecture.
    It consists of a transposed convolution layer (Upsampling) followed by a 
    convolutional block (Convolutions + ReLU).
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    skip_channels : int
        Number of skip connection channels.
    dropout_prob : float, optional
        Dropout probability, by default 0.2.
    use_checkpointing : bool, optional
        Whether to use gradient checkpointing, by default False.
    """
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, dropout_prob: float = 0.2, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        # Transposed convolution layer (Upsampling)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Dropout layer (applied after skip connection)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Main convolutional block (Convolutions + ReLU)
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Weight initialization he_normal (Kaiming Normal) for Conv layers
        for model_layer in self.conv_block.modules():
            if isinstance(model_layer, nn.Conv2d):
                nn.init.kaiming_normal_(model_layer.weight, nonlinearity='relu')
                if model_layer.bias is not None:
                    nn.init.constant_(model_layer.bias, 0)
        
        # Weight initialization he_normal (Kaiming Normal) for Transposed Conv
        nn.init.kaiming_normal_(self.upsample.weight, nonlinearity='relu')
        if self.upsample.bias is not None:
            nn.init.constant_(self.upsample.bias, 0)

    def forward(self, expansive_input: torch.Tensor, contractive_input: torch.Tensor = None) -> torch.Tensor:
        # Apply transposed convolution (Upsampling)
        up = self.upsample(expansive_input)
        # Dropout layer (applied after skip connection)
        up = self.dropout(up)
        
        if contractive_input is not None:
            # Concatenate the upsampled features with the skip connection from the corresponding DownsamplingBlock
            merge = torch.cat([up, contractive_input], dim=1).contiguous()
        else:
            merge = up.contiguous()
            
        # Convolutions + ReLU
        if self.use_checkpointing and self.training:
            conv = checkpoint(self.conv_block, merge, use_reentrant=False, preserve_rng_state=False)
        else:
            conv = self.conv_block(merge)
        
        return conv


class U_Net(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Parameters
    ----------
    filters : list[int] | None, optional
        List of filter sizes for the encoder blocks. If None, ResNet34 is used as the encoder, by default None.
    dropout_prob : float, optional
        Dropout probability for the decoder blocks, by default 0.2.
    in_channels : int, optional
        Number of input channels, by default 3.
    num_classes : int, optional
        Number of output classes, by default 19.
    use_checkpointing : bool | list[bool], optional
        Whether to use gradient checkpointing for each block, by default False.
    freeze_encoder : bool, optional
        Whether to freeze the encoder weights, by default True.
    """
    def __init__(self, filters: list[int] | None = None, dropout_prob: float = 0.2, in_channels: int = 3, 
                 num_classes: int = 19, use_checkpointing: bool | list[bool] = False, freeze_encoder: bool = True):
        super().__init__()
        
        self.use_resnet = filters is None
        filters_len = len(filters) if filters is not None else 5
        
        # Checkpointing list logic
        if isinstance(use_checkpointing, bool):
            use_checkpointing = [use_checkpointing] * filters_len
            
        if len(use_checkpointing) != filters_len:
            raise ValueError(f"use_checkpointing list must have the same length as filters ({filters_len})")
            
        self.use_checkpointing = any(use_checkpointing)

        if self.use_resnet:
            # ====================
            # Contracting Path (Encoder - ResNet34)
            # ====================
            resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
                
            self.encoder = nn.ModuleDict({
                'stem': nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu
                ), 
                'maxpool': resnet.maxpool,
                'layer1': resnet.layer1,
                'layer2': resnet.layer2,
                'layer3': resnet.layer3
            })

            # ====================
            # Bottleneck
            # ====================
            self.bottleneck = resnet.layer4
            
            # ====================
            # Expanding Path (Decoder - ResNet34)
            # ====================
            self.decoder = nn.ModuleList([
                # Up1: 512 -> upsample to 256 + 256 (skip3) -> 256
                UpsamplingBlock(in_channels=512, out_channels=256, skip_channels=256, dropout_prob=dropout_prob, 
                                    use_checkpointing=use_checkpointing[0]),
                # Up2: 256 -> upsample to 128 + 128 (skip2) -> 128
                UpsamplingBlock(in_channels=256, out_channels=128, skip_channels=128, dropout_prob=dropout_prob, 
                                    use_checkpointing=use_checkpointing[1]),
                # Up3: 128 -> upsample to 64 + 64 (skip1) -> 64
                UpsamplingBlock(in_channels=128, out_channels=64, skip_channels=64, dropout_prob=dropout_prob, 
                                    use_checkpointing=use_checkpointing[2]),
                # Up4: 64 -> upsample to 64 + 64 (skip0) -> 64
                UpsamplingBlock(in_channels=64, out_channels=64, skip_channels=64, dropout_prob=dropout_prob, 
                                use_checkpointing=use_checkpointing[3]),
                # Up5: 64 -> upsample to 32 (no skip) -> 32
                UpsamplingBlock(in_channels=64, out_channels=32, skip_channels=0, dropout_prob=dropout_prob, 
                                use_checkpointing=use_checkpointing[4]),
            ])
            self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)
            
        else:
            # ====================
            # Contracting Path (Encoder - Custom)
            # ====================
            current_channels = in_channels
            encoder_filters = filters[:-1]
            bottleneck_filter = filters[-1]
            
            self.encoder = nn.ModuleList()
            for layer, out_channels in enumerate(encoder_filters):
                self.encoder.append(
                    DownsamplingBlock(current_channels, out_channels, dropout_prob=dropout_prob, use_checkpointing=use_checkpointing[layer])
                )
                current_channels = out_channels
                
            # ====================
            # Bottleneck
            # ====================
            self.bottleneck = DownsamplingBlock(
                current_channels, bottleneck_filter, dropout_prob=0, max_pooling=False, use_checkpointing=use_checkpointing[-1]
            )
            current_channels = bottleneck_filter
            
            # ====================
            # Expanding Path (Decoder - Custom)
            # ====================
            self.decoder = nn.ModuleList()
            for layer, out_channels in enumerate(reversed(encoder_filters)):
                self.decoder.append(
                    UpsamplingBlock(current_channels, out_channels, skip_channels=out_channels,
                    dropout_prob=dropout_prob, use_checkpointing=use_checkpointing[-(layer + 2)])
                )
                current_channels = out_channels

            self.output_conv = nn.Conv2d(current_channels, num_classes, kernel_size=1)

        # ====================
        # Freeze Encoder (Transfer Learning / Fine-Tuning)
        # ====================
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.bottleneck.parameters():
                param.requires_grad = False

    @torch.compiler.disable
    def _eager_maxpool(self, x):
        """Executes MaxPool2d outside of torch.compile to prevent a torchinductor bug."""
        return self.encoder['maxpool'](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch checkpointing requires at least one input to have requires_grad=True 
        if self.use_checkpointing and self.training and not x.requires_grad:
            x.requires_grad_(True)
            
        if self.use_resnet:
            # ====================
            # Encoder (ResNet34)
            # ====================
            skip0 = self.encoder['stem'](x)         # H/2, W/2, 64 channels
            
            x = self._eager_maxpool(skip0)          # H/4, W/4, 64 channels
            skip1 = self.encoder['layer1'](x)       # H/4, W/4, 64 channels
            
            skip2 = self.encoder['layer2'](skip1)   # H/8, W/8, 128 channels
            skip3 = self.encoder['layer3'](skip2)   # H/16, W/16, 256 channels
            
            bottleneck = self.bottleneck(skip3)     # H/32, W/32, 512 channels
            
            # ====================
            # Decoder
            # ====================
            x = bottleneck
            x = self.decoder[0](x, skip3)        # Up 1
            x = self.decoder[1](x, skip2)        # Up 2
            x = self.decoder[2](x, skip1)        # Up 3
            x = self.decoder[3](x, skip0)        # Up 4
            x = self.decoder[4](x, None)         # Up 5 (No skip connection)
            
        else:
            skip_connections = []
            
            # ====================
            # Encoder (Custom)
            # ====================
            for encoder_block in self.encoder:
                x, skip = encoder_block(x)
                skip_connections.append(skip)
                
            # Bottleneck
            _, x = self.bottleneck(x)
            
            # Reverse the skip connections list
            skip_connections = skip_connections[::-1]
            
            # ====================
            # Decoder
            # ====================
            for skip, decoder_block in zip(skip_connections, self.decoder):
                x = decoder_block(x, skip)
                
        # ====================
        # Final output
        # ====================
        output = self.output_conv(x)
        
        return output