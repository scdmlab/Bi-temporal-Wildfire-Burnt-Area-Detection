"""
Bi-temporal Attention U-Net for Wildfire Burnt Area Detection

Author: Tang Sui
Email: tsui5@wisc.edu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f'Initialize network with {init_type}')
    net.apply(init_func)


class ConvBlock(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upsampling convolution block"""
    def __init__(self, in_channels, out_channels, use_transpose=True):
        super(UpConv, self).__init__()
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate mechanism"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net for single image segmentation"""
    def __init__(self, in_channels=3, num_classes=1, channels=[64, 128, 256, 512, 1024], 
                 use_attention=True, use_transpose=True, init_weights_flag=True):
        super(AttentionUNet, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1])
        self.conv3 = ConvBlock(channels[1], channels[2])
        self.conv4 = ConvBlock(channels[2], channels[3])
        self.conv5 = ConvBlock(channels[3], channels[4])
        
        # Decoder
        self.up5 = UpConv(channels[4], channels[3], use_transpose)
        self.up_conv5 = ConvBlock(channels[4], channels[3])
        
        self.up4 = UpConv(channels[3], channels[2], use_transpose)
        self.up_conv4 = ConvBlock(channels[3], channels[2])
        
        self.up3 = UpConv(channels[2], channels[1], use_transpose)
        self.up_conv3 = ConvBlock(channels[2], channels[1])
        
        self.up2 = UpConv(channels[1], channels[0], use_transpose)
        self.up_conv2 = ConvBlock(channels[1], channels[0])
        
        # Attention gates
        if use_attention:
            self.att5 = AttentionGate(channels[3], channels[3], channels[2])
            self.att4 = AttentionGate(channels[2], channels[2], channels[1])
            self.att3 = AttentionGate(channels[1], channels[1], 64)
            self.att2 = AttentionGate(channels[0], channels[0], channels[0] // 2)
        
        # Output layer
        self.conv_1x1 = nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0)
        
        if init_weights_flag:
            init_weights(self)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        
        # Decoder with skip connections
        d5 = self.up5(x5)
        if self.use_attention:
            x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)
        
        d4 = self.up4(d5)
        if self.use_attention:
            x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)
        
        d3 = self.up3(d4)
        if self.use_attention:
            x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)
        
        d2 = self.up2(d3)
        if self.use_attention:
            x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        
        # Output
        output = self.conv_1x1(d2)
        return output


def crop_tensor(tensor, target_tensor):
    """Crop tensor to match target tensor size"""
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class BiTemporalUNet(nn.Module):
    """Bi-temporal U-Net for wildfire burnt area detection"""
    def __init__(self, in_channels=3, num_classes=1, channels=[64, 128, 256, 512, 1024], 
                 use_transpose=True, init_weights_flag=True):
        super(BiTemporalUNet, self).__init__()
        
        self.num_classes = num_classes
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shared encoder for both temporal images
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1])
        self.conv3 = ConvBlock(channels[1], channels[2])
        self.conv4 = ConvBlock(channels[2], channels[3])
        self.conv5 = ConvBlock(channels[3], channels[4])
        
        # Decoder for fused features
        self.up_trans_2048 = nn.ConvTranspose2d(2048, 1024, kernel_size=5, stride=2, padding=2)
        self.up_conv_2048_1024 = ConvBlock(2048, 1024)
        
        self.up_trans_1024 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2)
        self.up_conv_1024_512 = ConvBlock(1024, 512)
        
        self.up_trans_512 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2)
        self.up_conv_512_256 = ConvBlock(512, 256)
        
        self.up_trans_256 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2)
        self.up_conv_256_128 = ConvBlock(256, 128)
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )
        
        if init_weights_flag:
            init_weights(self)

    def encode(self, x):
        """Encode single image"""
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        return x1, x2, x3, x4, x5

    def forward(self, x_pre, x_post):
        # Encode both temporal images
        x1_pre, x2_pre, x3_pre, x4_pre, x5_pre = self.encode(x_pre)
        x1_post, x2_post, x3_post, x4_post, x5_post = self.encode(x_post)
        
        # Feature fusion at deepest level
        bridge = torch.cat((x5_pre, x5_post), dim=1)
        
        # Decoder with feature fusion at each level
        d5 = self.up_trans_2048(bridge)
        d5_crop = crop_tensor(d5, x4_pre)
        d5_cat = torch.cat((d5_crop, x4_pre, x4_post), dim=1)
        d5 = self.up_conv_2048_1024(d5_cat)
        
        d4 = self.up_trans_1024(d5)
        d4_crop = crop_tensor(d4, x3_pre)
        d4_cat = torch.cat((d4_crop, x3_pre, x3_post), dim=1)
        d4 = self.up_conv_1024_512(d4_cat)
        
        d3 = self.up_trans_512(d4)
        d3_crop = crop_tensor(d3, x2_pre)
        d3_cat = torch.cat((d3_crop, x2_pre, x2_post), dim=1)
        d3 = self.up_conv_512_256(d3_cat)
        
        d2 = self.up_trans_256(d3)
        d2_crop = crop_tensor(d2, x1_pre)
        d2_cat = torch.cat((d2_crop, x1_pre, x1_post), dim=1)
        d2 = self.up_conv_256_128(d2_cat)
        
        # Output
        output = self.output_conv(d2)
        return output


class BiTemporalAttentionUNet(nn.Module):
    """Bi-temporal Attention U-Net for wildfire burnt area detection"""
    def __init__(self, in_channels=3, num_classes=1, channels=[64, 128, 256, 512, 1024], 
                 use_transpose=True, init_weights_flag=True):
        super(BiTemporalAttentionUNet, self).__init__()
        
        self.num_classes = num_classes
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shared encoder
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1])
        self.conv3 = ConvBlock(channels[1], channels[2])
        self.conv4 = ConvBlock(channels[2], channels[3])
        self.conv5 = ConvBlock(channels[3], channels[4])
        
        # Attention gates for feature fusion
        self.att5 = AttentionGate(channels[3], channels[3]*2, channels[2])
        self.att4 = AttentionGate(channels[2], channels[2]*2, channels[1])
        self.att3 = AttentionGate(channels[1], channels[1]*2, 64)
        self.att2 = AttentionGate(channels[0], channels[0]*2, channels[0]//2)
        
        # Decoder
        self.up_trans_2048 = nn.ConvTranspose2d(2048, 1024, kernel_size=5, stride=2, padding=2)
        self.up_conv_2048_1024 = ConvBlock(1024 + channels[3], 1024)
        
        self.up_trans_1024 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2)
        self.up_conv_1024_512 = ConvBlock(512 + channels[2], 512)
        
        self.up_trans_512 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2)
        self.up_conv_512_256 = ConvBlock(256 + channels[1], 256)
        
        self.up_trans_256 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2)
        self.up_conv_256_128 = ConvBlock(128 + channels[0], 128)
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )
        
        if init_weights_flag:
            init_weights(self)

    def encode(self, x):
        """Encode single image"""
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        return x1, x2, x3, x4, x5

    def forward(self, x_pre, x_post):
        # Encode both temporal images
        x1_pre, x2_pre, x3_pre, x4_pre, x5_pre = self.encode(x_pre)
        x1_post, x2_post, x3_post, x4_post, x5_post = self.encode(x_post)
        
        # Feature fusion at deepest level
        bridge = torch.cat((x5_pre, x5_post), dim=1)
        
        # Decoder with attention-guided feature fusion
        d5 = self.up_trans_2048(bridge)
        d5_crop = crop_tensor(d5, x4_pre)
        x4_fused = torch.cat((x4_pre, x4_post), dim=1)
        x4_att = self.att5(g=d5_crop, x=x4_fused)
        d5 = self.up_conv_2048_1024(torch.cat((d5_crop, x4_att), dim=1))
        
        d4 = self.up_trans_1024(d5)
        d4_crop = crop_tensor(d4, x3_pre)
        x3_fused = torch.cat((x3_pre, x3_post), dim=1)
        x3_att = self.att4(g=d4_crop, x=x3_fused)
        d4 = self.up_conv_1024_512(torch.cat((d4_crop, x3_att), dim=1))
        
        d3 = self.up_trans_512(d4)
        d3_crop = crop_tensor(d3, x2_pre)
        x2_fused = torch.cat((x2_pre, x2_post), dim=1)
        x2_att = self.att3(g=d3_crop, x=x2_fused)
        d3 = self.up_conv_512_256(torch.cat((d3_crop, x2_att), dim=1))
        
        d2 = self.up_trans_256(d3)
        d2_crop = crop_tensor(d2, x1_pre)
        x1_fused = torch.cat((x1_pre, x1_post), dim=1)
        x1_att = self.att2(g=d2_crop, x=x1_fused)
        d2 = self.up_conv_256_128(torch.cat((d2_crop, x1_att), dim=1))
        
        # Output
        output = self.output_conv(d2)
        return output


# Model factory function
def get_model(model_type='bitemporal_unet', **kwargs):
    """
    Factory function to get model instance
    
    Args:
        model_type (str): Type of model ('unet', 'attention_unet', 'bitemporal_unet', 'bitemporal_attention_unet')
        **kwargs: Model parameters
    
    Returns:
        torch.nn.Module: Model instance
    """
    models = {
        'unet': lambda: AttentionUNet(use_attention=False, **kwargs),
        'attention_unet': lambda: AttentionUNet(use_attention=True, **kwargs),
        'bitemporal_unet': lambda: BiTemporalUNet(**kwargs),
        'bitemporal_attention_unet': lambda: BiTemporalAttentionUNet(**kwargs)
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type]()


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test BiTemporal UNet
    model = get_model('bitemporal_unet', num_classes=1)
    model.to(device)
    
    # Test forward pass
    x_pre = torch.randn(2, 3, 224, 224).to(device)
    x_post = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(x_pre, x_post)
        print(f"Model output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
