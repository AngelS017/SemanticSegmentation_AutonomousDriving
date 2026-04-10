import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


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
        Dropout probability, by default 0.0.
    max_pooling : bool, optional
        Whether to apply max pooling, by default True.
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.2, max_pooling=True, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        # Main convolutional block (Convolutions + ReLU + Dropout)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
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
        if self.use_checkpointing:
            skip_connection = checkpoint(self.conv_block, x, use_reentrant=False)
        else:
            skip_connection = self.conv_block(x)
        
        # 2. Max Pooling
        next_layer = self.maxpool(skip_connection)

        # 3. Dropout
        next_layer = self.dropout(next_layer)

        return next_layer, skip_connection


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
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.2, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        # Transposed convolution layer (Upsampling)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Dropout layer (applied after skip connection)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Main convolutional block (Convolutions + ReLU)
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
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

    def forward(self, expansive_input, contractive_input):
        # Apply transposed convolution (Upsampling)
        up = self.upsample(expansive_input)
        # Dropout layer (applied after skip connection)
        up = self.dropout(up)
        # Concatenate the upsampled features with the skip connection from the corresponding DownsamplingBlock 
        merge = torch.cat([up, contractive_input], dim=1)
        # Convolutions + ReLU
        if self.use_checkpointing:
            conv = checkpoint(self.conv_block, merge, use_reentrant=False)
        else:
            conv = self.conv_block(merge)
        
        return conv


class U_Net(nn.Module):
    def __init__(self, filters, dropout_prob, in_channels=3, num_classes=19, use_checkpointing=False):
        super().__init__()
        
        # Checkpointing list logic: Can be bool or list of len(filters) mapping to each depth level
        if isinstance(use_checkpointing, bool):
            use_checkpointing = [use_checkpointing] * len(filters)
            
        if len(use_checkpointing) != len(filters):
            raise ValueError(f"use_checkpointing list must have the same length as filters ({len(filters)})")
            
        self.use_checkpointing = any(use_checkpointing)

        current_channels = in_channels
        
        encoder_filters = filters[:-1]
        bottleneck_filter = filters[-1]
        
        # ====================
        # Contracting Path (Encoder)
        # ====================
        self.encoder = nn.ModuleList()
        for layer, out_channels in enumerate(encoder_filters):
            # Apply Dropout only in the last downsampling layer
            curr_dropout = dropout_prob if layer == len(encoder_filters) - 1 else 0.0
            
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
        # Expanding Path (Decoder)
        # ====================
        self.decoder = nn.ModuleList()
        
        for layer, out_channels in enumerate(reversed(encoder_filters)):
            self.decoder.append(
                UpsamplingBlock(current_channels, out_channels, dropout_prob=dropout_prob, use_checkpointing=use_checkpointing[-(layer + 2)])
            )
            current_channels = out_channels

        # ====================
        # Pre-output Refinement Conv
        # Extra 3x3 conv at full resolution to refine features before classification
        # ====================
        self.pre_output_conv = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.pre_output_conv[0].weight, nonlinearity='relu')
        nn.init.constant_(self.pre_output_conv[0].bias, 0)

        # ====================
        # Output (1x1 Convolution)
        # ====================
        self.output_conv = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def forward(self, x):
        if self.use_checkpointing and getattr(x, 'requires_grad', False) is False:
            x.requires_grad_(True)
            
        skip_connections = []
        
        # Encoder (accumulate skip connections)
        for encoder_block in self.encoder:
            x, skip = encoder_block(x)
            skip_connections.append(skip)
            
        # Bottleneck (next_layer and skip are the same here since max_pooling=False, we keep x)
        _, x = self.bottleneck(x)
        
        # Reverse the skip connections list (deepest encoder skip matches first decoder block)
        skip_connections = skip_connections[::-1]
        
        # Decoder (pair each skip connection with its corresponding decoder block)
        for skip, decoder_block in zip(skip_connections, self.decoder):
            x = decoder_block(x, skip)
            
        # Final refinement at full resolution before classification
        x = self.pre_output_conv(x)
        
        # Final output (pixel-wise classification)
        output = self.output_conv(x)
        
        return output