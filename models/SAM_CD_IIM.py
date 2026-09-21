import torch
from torch import nn
from torch.nn import functional as F
from utils.misc import initialize_weights
from lib.pvtv2 import pvt_v2_b2

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()        
        A = self.SA(x)
        return A

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)        
        return x

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

##adapter
class MSA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSA, self).__init__()
        self.convs =  BasicConv2d(in_channel, out_channel, 1)
        self.c0 = BasicConv2d(out_channel, out_channel, 3,1,1)
        self.c1 = BasicConv2d(out_channel, out_channel, 5, 1, 2)
        self.c2 = BasicConv2d(out_channel, out_channel, 7, 1, 3)
        self.c3 = BasicConv2d(out_channel, out_channel, 9, 1, 4)
        self.c4 = BasicConv2d(out_channel*4,out_channel,1)
        self.cbam = CBAM(out_channel)

    def forward(self, features):
        features = self.convs(features)
        features0 = self.c0(features)
        features1 = self.c1(features)
        features2 = self.c2(features)
        features3 = self.c3(features)
        features4 = self.c4(torch.cat([features0, features1,features2,features3],1))
        out = self.cbam(features+features4)
        return out

class BAM(nn.Module):
    def __init__(self, out_channel):
        super(BAM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv22 = BasicConv2d(out_channel, out_channel, 3, padding=1)

    def forward(self,mask,boundary):
        ##edge semantics
        edge = self.sigmoid(boundary)
        x = edge*mask
        x = self.conv22(mask + x)
        return x


##
from Dinov3.engine.backbone.dinov3_adapter import DINOv3STAs
class DMBNet(nn.Module):
    def __init__(
        self,
        num_embed=8,
        pretrained_path=r'',
        weight_path=r'',
        drop_rate=0.2,
        ):
        super(DMBNet, self).__init__()

        self.dinov3_model = DINOv3STAs(weights_path=weight_path)

        self.segmenter = nn.Conv2d(64, num_embed, kernel_size=1)

        ##PVT
        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        save_model = torch.load(pretrained_path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Adapter128 = MSA(64,40)
        self.Adapter64 = MSA(128,80)
        self.Adapter32 = MSA(320,160)
        self.Adapter16 = MSA(512,160)

        self.conv128 = BasicConv2d(256,64,1)
        self.conv64 = BasicConv2d(256,128,1)
        self.conv32 = BasicConv2d(256,320,1)
        self.conv16 = BasicConv2d(256,512,1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv1 = BasicConv2d(320,160,1)
        self.conv2 = BasicConv2d(240,128,1)
        self.conv3 = BasicConv2d(168,64,1)
        self.conv4 = BasicConv2d(128,64,1)
        self.conv5 = BasicConv2d(128,64,1)

        self.bam = BAM(64)
        self.segmenterC = nn.Conv2d(64, 1, kernel_size=1)
        self.segmenterD = nn.Conv2d(64, 1, kernel_size=1)
        #
        initialize_weights(self.Adapter128, self.Adapter64, self.Adapter32, self.Adapter16, \
                           self.segmenter, self.conv1, self.conv2, self.segmenterC, self.segmenterD, self.conv3, self.conv4, \
                           self.conv5)

    def run_encoder(self, image):
        feats1, feats2, feats3, feats4 =  self.dinov3_model(image)
        ##PVT-V2
        pvt = self.backbone(image)
        x1 = pvt[0]              ##64x128X128
        x2 = pvt[1]             ##128x64X64
        x3 = pvt[2]               ##320x32X32
        x4 = pvt[3]               ##512x16X16

        ##联合再使用多尺度适配器
        M1 = self.Adapter128(feats1+x1)  ##40x128X128
        M2 = self.Adapter64(feats2+x2)   ##80xx64X64
        M3 = self.Adapter32(feats3+x3)   ##160x32X32
        M4 = self.Adapter16(feats4+x4)  ##160x16X16

        return M1, M2, M3, M4

    def run_decoder(self, x1, x2, x3, x4):
        ##
        s1 = self.conv1(torch.cat([self.up1(x4),x3],1))
        s2 = self.conv2(torch.cat([self.up1(s1),x2],1))
        s3 = self.conv3(torch.cat([self.up1(s2),x1],1)) ##64x128x128

        return s3

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        M11, M22, M33, M44 = self.run_encoder(x1)
        feaA = self.run_decoder(M11, M22, M33, M44)
        outA = self.segmenter(feaA)
        Mt1, Mt2, Mt3, Mt4 = self.run_encoder(x2)
        feaB = self.run_decoder(Mt1, Mt2, Mt3, Mt4)
        outB = self.segmenter(feaB)

        mask = self.conv4(torch.cat([feaA,feaB],1))
        boundary = self.conv5(torch.cat([feaA,feaB],1))

        outC = self.segmenterC(self.bam(mask,boundary))
        out_nd = self.segmenterD(boundary)

        outC = F.interpolate(outC, scale_factor=4, mode='bilinear')
        outA = F.interpolate(outA, scale_factor=4, mode='bilinear')
        outB = F.interpolate(outB, scale_factor=4, mode='bilinear')
        out_nd = F.interpolate(out_nd, scale_factor=4, mode='bilinear')

        return outC, outA, outB, out_nd

