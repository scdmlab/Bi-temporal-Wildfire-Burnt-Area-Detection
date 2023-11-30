import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch_out,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, convTranspose=True):
        super(up_conv, self).__init__()
        if convTranspose:
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_in, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2)

        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        x = self.Conv(x)
        return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        torch.save(psi.to(torch.device('cpu')), "psi_50.pth")
        return x * psi


class U_Net(nn.Module):

    def __init__(self,
                 in_channel=3,
                 num_classes=1,
                 channel_list=[64, 128, 256, 512, 1024],
                 checkpoint=False,
                 convTranspose=True):
        super(U_Net, self).__init__()
        self.num_classes = num_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channel, ch_out=channel_list[0])
        self.Conv2 = conv_block(ch_in=channel_list[0], ch_out=channel_list[1])
        self.Conv3 = conv_block(ch_in=channel_list[1], ch_out=channel_list[2])
        self.Conv4 = conv_block(ch_in=channel_list[2], ch_out=channel_list[3])
        self.Conv5 = conv_block(ch_in=channel_list[3], ch_out=channel_list[4])

        self.Up5 = up_conv(ch_in=channel_list[4], ch_out=channel_list[3], convTranspose=convTranspose)
        self.Up_conv5 = conv_block(ch_in=channel_list[4],
                                   ch_out=channel_list[3])

        self.Up4 = up_conv(ch_in=channel_list[3], ch_out=channel_list[2], convTranspose=convTranspose)
        self.Up_conv4 = conv_block(ch_in=channel_list[3],
                                   ch_out=channel_list[2])

        self.Up3 = up_conv(ch_in=channel_list[2], ch_out=channel_list[1], convTranspose=convTranspose)
        self.Up_conv3 = conv_block(ch_in=channel_list[2],
                                   ch_out=channel_list[1])

        self.Up2 = up_conv(ch_in=channel_list[1], ch_out=channel_list[0], convTranspose=convTranspose)
        self.Up_conv2 = conv_block(ch_in=channel_list[1],
                                   ch_out=channel_list[0])

        self.Conv_1x1 = nn.Conv2d(channel_list[0],
                                  num_classes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

        if not checkpoint:
            init_weights(self)

    def forward(self, x):
        # encoder
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoder
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self,
                 in_channel=3,
                 num_classes=1,
                 channel_list=[64, 128, 256, 512, 1024],
                 checkpoint=False,
                 convTranspose=True):
        super(AttU_Net, self).__init__()
        self.num_classes = num_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channel, ch_out=channel_list[0])
        self.Conv2 = conv_block(ch_in=channel_list[0], ch_out=channel_list[1])
        self.Conv3 = conv_block(ch_in=channel_list[1], ch_out=channel_list[2])
        self.Conv4 = conv_block(ch_in=channel_list[2], ch_out=channel_list[3])
        self.Conv5 = conv_block(ch_in=channel_list[3], ch_out=channel_list[4])

        self.Up5 = up_conv(ch_in=channel_list[4], ch_out=channel_list[3], convTranspose=convTranspose)
        self.Att5 = Attention_block(F_g=channel_list[3],
                                    F_l=channel_list[3],
                                    F_int=channel_list[2])
        self.Up_conv5 = conv_block(ch_in=channel_list[4],
                                   ch_out=channel_list[3])

        self.Up4 = up_conv(ch_in=channel_list[3], ch_out=channel_list[2], convTranspose=convTranspose)
        self.Att4 = Attention_block(F_g=channel_list[2],
                                    F_l=channel_list[2],
                                    F_int=channel_list[1])
        self.Up_conv4 = conv_block(ch_in=channel_list[3],
                                   ch_out=channel_list[2])

        self.Up3 = up_conv(ch_in=channel_list[2], ch_out=channel_list[1], convTranspose=convTranspose)
        self.Att3 = Attention_block(F_g=channel_list[1],
                                    F_l=channel_list[1],
                                    F_int=64)
        self.Up_conv3 = conv_block(ch_in=channel_list[2],
                                   ch_out=channel_list[1])

        self.Up2 = up_conv(ch_in=channel_list[1], ch_out=channel_list[0], convTranspose=convTranspose)
        self.Att2 = Attention_block(F_g=channel_list[0],
                                    F_l=channel_list[0],
                                    F_int=channel_list[0] // 2)
        self.Up_conv2 = conv_block(ch_in=channel_list[1],
                                   ch_out=channel_list[0])

        self.Conv_1x1 = nn.Conv2d(channel_list[0],
                                  num_classes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)


        if not checkpoint:
            init_weights(self)

    def forward(self, x):
        # encoder
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoder
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



'''
Bi-temporal
'''


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]  # 0=batchsize 1=channel 2=height 3=width
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size  # assuming tensor size always > target size (crop)
    delta = delta / 2
    #         return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
    if delta.is_integer():
        return tensor[:, :, int(delta):tensor_size - (int(delta)), int(delta):tensor_size - (int(delta))]
    return tensor[:, :, int(delta):tensor_size - (int(delta) + 1), int(delta):tensor_size - (int(delta) + 1)]


class double_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1, bias=True, groups=1),
            nn.BatchNorm2d(num_features=out_c),
            # nn.GroupNorm(num_groups=out_c//16, num_channels=out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1, bias=True, groups=1),
            nn.BatchNorm2d(num_features=out_c),
            # nn.GroupNorm(num_groups=out_c//16, num_channels=out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class BridgeUNet(nn.Module):
    def __init__(self):
        super(BridgeUNet, self).__init__()

        # 2x2 max pool stride 2
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # layers based on the double convolution
        self.down_conv_4_64 = double_conv(4, 64)  # number of input channels (in = 4, out = 64)
        self.down_conv_64_128 = double_conv(64, 128)  # in = 64, out = 128
        self.down_conv_128_256 = double_conv(128, 256)
        self.down_conv_256_512 = double_conv(256, 512)
        self.down_conv_512_1024 = double_conv(512, 1024)
        # after this part -> go to forward -> encoder

    def forward(self, image):
        # forward pass: expected size = batch size, channel, height, width
        # encoder
        x1d = self.down_conv_4_64(image)  # 1 output is also passed to second part of network
        # print('x1d: ', x1d.size())

        x1dm = self.max_pool_2x2(x1d)
        x2d = self.down_conv_64_128(x1dm)  # 2 output is also passed to second part of network
        # print('x2d: ', x2d.size())

        x2dm = self.max_pool_2x2(x2d)
        x3d = self.down_conv_128_256(x2dm)  # 3 output is also passed to second part of network
        # print('x3d: ', x3d.size())

        x3dm = self.max_pool_2x2(x3d)
        x4d = self.down_conv_256_512(x3dm)  # 4 output is also passed to second part of network
        # print('x4d: ', x4d.size())

        x4dm = self.max_pool_2x2(x4d)
        x5d = self.down_conv_512_1024(x4dm)
        # print('x5d: ', x5d.size())

        return x5d, x4d, x3d, x2d, x1d


class out_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out_conv, self).__init__()
        self.out = nn.Conv2d(in_channels=128, out_channels=1, groups=1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.out(x)
        return torch.sigmoid(x)


class Bi_U_Net(nn.Module):

    def __init__(self,
                 in_channel=3,
                 num_classes=1,
                 channel_list=[64, 128, 256, 512, 1024],
                 checkpoint=False,
                 convTranspose=True):
        super(Bi_U_Net, self).__init__()
        self.up_trans_2048 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=5, stride=2, padding=0)
        self.up_conv_2048_1024 = double_conv(2048, 1024)

        self.up_trans_1024 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=0)
        self.up_conv_1024_512 = double_conv(1024, 512)

        self.up_trans_512 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=0)
        self.up_conv_512_256 = double_conv(512, 256)

        self.up_trans_256 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=0)
        self.up_conv_256_128 = double_conv(256, 128)
        self.num_classes = num_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channel, ch_out=channel_list[0])
        self.Conv2 = conv_block(ch_in=channel_list[0], ch_out=channel_list[1])
        self.Conv3 = conv_block(ch_in=channel_list[1], ch_out=channel_list[2])
        self.Conv4 = conv_block(ch_in=channel_list[2], ch_out=channel_list[3])
        self.Conv5 = conv_block(ch_in=channel_list[3], ch_out=channel_list[4])

        self.Up5 = up_conv(ch_in=channel_list[4], ch_out=channel_list[3], convTranspose=convTranspose)
        self.Up_conv5 = conv_block(ch_in=channel_list[4],
                                   ch_out=channel_list[3])

        self.Up4 = up_conv(ch_in=channel_list[3], ch_out=channel_list[2], convTranspose=convTranspose)
        self.Up_conv4 = conv_block(ch_in=channel_list[3],
                                   ch_out=channel_list[2])

        self.Up3 = up_conv(ch_in=channel_list[2], ch_out=channel_list[1], convTranspose=convTranspose)
        self.Up_conv3 = conv_block(ch_in=channel_list[2],
                                   ch_out=channel_list[1])

        self.Up2 = up_conv(ch_in=channel_list[1], ch_out=channel_list[0], convTranspose=convTranspose)
        self.Up_conv2 = conv_block(ch_in=channel_list[1],
                                   ch_out=channel_list[0])

        self.Conv_1x1 = nn.Conv2d(channel_list[0],
                                  num_classes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

        # output layer
        self.out_conv = out_conv(128, 1)

        if not checkpoint:
            init_weights(self)

    def forward(self, x_pre, x_post):
        # encoder
        x1_pre = self.Conv1(x_pre)

        x2_pre = self.Maxpool(x1_pre)
        x2_pre = self.Conv2(x2_pre)

        x3_pre = self.Maxpool(x2_pre)
        x3_pre = self.Conv3(x3_pre)

        x4_pre = self.Maxpool(x3_pre)
        x4_pre = self.Conv4(x4_pre)

        x5_pre = self.Maxpool(x4_pre)
        x5_pre = self.Conv5(x5_pre)

        x1_post = self.Conv1(x_post)

        x2_post = self.Maxpool(x1_post)
        x2_post = self.Conv2(x2_post)

        x3_post = self.Maxpool(x2_post)
        x3_post = self.Conv3(x3_post)

        x4_post = self.Maxpool(x3_post)
        x4_post = self.Conv4(x4_post)

        x5_post = self.Maxpool(x4_post)
        x5_post = self.Conv5(x5_post)

        bridge = torch.cat((x5_post, x5_pre), dim=1)

        # decoder

        dB_B = self.up_trans_2048(bridge)
        dB_crop = crop_img(dB_B, x4_pre)
        dB_u = torch.cat((dB_crop, x4_pre, x4_post), dim=1)
        dB = self.up_conv_2048_1024(dB_u)

        d4_B = self.up_trans_1024(dB)
        d4_crop = crop_img(d4_B, x3_pre)
        d4_U = torch.cat((d4_crop, x3_pre, x3_post), dim=1)
        d4 = self.up_conv_1024_512(d4_U)

        d3_B = self.up_trans_512(d4)
        d3_crop = crop_img(d3_B, x2_pre)
        d3_U = torch.cat((d3_crop, x2_pre, x2_post), dim=1)
        d3 = self.up_conv_512_256(d3_U)

        d2_B = self.up_trans_256(d3)
        d2_crop = crop_img(d2_B, x1_pre)
        d2_U = torch.cat((d2_crop, x1_pre, x1_post), dim=1)
        d2 = self.up_conv_256_128(d2_U)

        d1 = self.out_conv(d2)

        return d1
