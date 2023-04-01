import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import functools

class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet34(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)         # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        
        
        return feature2, feature3, feature4

class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1,2,3,6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
            feat  = feat + x
        return feat

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel,  out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.left(x)
        residual = self.shortcut(x)
        out += residual
        return F.relu(out)


class FeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, in_channel, out_channel):
        super(FeatureFusion, self).__init__()
        self.fusion = ResBlock(in_channel, out_channel)
        
    def forward(self, x_high, x_low):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_low, x_high), dim=1)
        x = self.fusion(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Detail_path(nn.Module):
    def __init__(self):
        super(Detail_path, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(3,  32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32,  64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(3,  128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128)
        )

    def forward(self, x):
        out = self.left(x)
        residual = self.shortcut(x)

        out += residual
        return F.relu(out)


class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        # self.relu = nn.ReLU(inplace = True)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y_1 = self.conv(x)
        y_1 = self.bn(y_1)
        y_1 = self.relu(y_1)

        return y_1

class projectors(nn.Module):
    def __init__(self, input_nc=1, ndf=8, norm_layer=nn.BatchNorm2d):
        super(projectors, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(input_nc, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
    def forward(self, input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_out = self.conv_2(x_0)
        x_out = self.pool(x_out)
        x_out = self.final(x_out)
        return x_out

class classifier(nn.Module):
    def __init__(self, inp_dim=1, ndf=8, norm_layer=nn.BatchNorm2d):
        super(classifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(inp_dim, ndf)
        self.conv_2 = conv(ndf, ndf * 2)
        self.conv_3 = conv(ndf * 2, ndf * 4)
        self.final = nn.Conv2d(ndf * 4, ndf * 4, kernel_size=1)
        # self.linear = nn.Linear(in_features=ndf*4*18*12, out_features=1024)

    def forward(self, input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_1 = self.conv_2(x_0)
        x_1 = self.pool(x_1)
        x_2 = self.conv_3(x_1)
        x_2 = self.pool(x_2)
        # x_out = self.linear(x_2)
        x_out = self.final(x_2)
        return x_out

class csdNet(nn.Module):
    """Image Cascade Network"""
    def __init__(self, nclass = 1):
        super(csdNet, self).__init__()
        self.nclass = nclass
        
        self.Morphology_path = resnet18(pretrained=True)
        
        self.headpool = PyramidPoolingModule()
        self.fusion1 = FeatureFusion(in_channel=512+256, out_channel=256)
        self.fusion2 = FeatureFusion(in_channel=256+128, out_channel=128)
        self.fusion3 = FeatureFusion(in_channel=128+128, out_channel=64)

        self.ca = ChannelAttention(128)
        self.sa = SpatialAttention()

        self.Detail_path = Detail_path()
        
        self.conv_cls_spa = nn.Conv2d(512, nclass, 1, 1, bias=False)              
        self.conv_cls_cnt = nn.Conv2d(128, nclass, 1, 1, bias=False)
        self.conv_cls_out = nn.Conv2d(64 , nclass, 1, 1, bias=False)
        
        
    def forward(self, x):

        # sub 1 Morphology_path
        f_c1, f_c2, f_c3 = self.Morphology_path(x)
        # print('00',f_c1.shape, f_c2.shape, f_c3.shape)
        f_c3 = self.headpool(f_c3)
        # print('0', f_c3.shape)
        

        f_f23 = self.fusion1(f_c2, f_c3)
        # print('1', f_f23.shape)
        f_c = self.fusion2(f_c1, f_f23)
        # print('2', f_c.shape)
        f_c = self.ca(f_c) * f_c
        # print('3', f_c.shape)
        f_c = self.sa(f_c) * f_c
        # print('4', f_c.shape)
        
        # sub 2 Detail _path
        f_e = self.Detail_path(x)
        # print('5', f_e.shape)

        f_o = self.fusion3(f_e, f_c)
        # print('7', f_o.shape)

        out = F.interpolate(f_o, size=x.size()[2:], mode='bilinear', align_corners=True)
        # print('8', out.shape)

        out = self.conv_cls_out(out)
        # print('9', out.shape)


        return out, torch.sigmoid(out)



