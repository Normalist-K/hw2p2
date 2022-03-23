import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        self.expansion = expansion
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.main_stream = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, 
                bias=False),
            nn.BatchNorm2d(self.out_ch),
            nn.GELU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, 
                stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_ch),
            nn.GELU(),
            nn.Conv2d(self.out_ch, self.out_ch*self.expansion, 
                kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_ch*self.expansion)
        )

        if self.stride != 1 or self.in_ch != self.out_ch*self.expansion:
            if stride == 2:
                self.skip = nn.Sequential(
                    nn.AvgPool2d((2,2), stride=2),
                    nn.Conv2d(self.in_ch, self.out_ch*self.expansion, 
                        kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.out_ch*self.expansion)
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(self.in_ch, self.out_ch*self.expansion, 
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.out_ch*self.expansion)
                )
        else:
            self.skip = nn.Identity()

        self.last_act = nn.GELU()
    
    def zero_init_last_bn(self):
        nn.init.zeros_(self.main_stream[-1].weight)

    def forward(self, input):
        x = self.main_stream(input)
        x = x + self.skip(input)
        out = self.last_act(x)

        return out

class ResNet_Variant(nn.Module):
    def __init__(self, num_blocks=[3, 4, 6, 3], num_classes=7000):
        super().__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.expansion = 4
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3,
                stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3,
                stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feat_layer = nn.Sequential(
            nn.Linear(512*self.expansion, 768)
        )
        self.cls_layer = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Linear(768, self.num_classes)
        )

        self.blocks1 = self.make_blocks(64, num_blocks[0], stride=1)
        self.blocks2 = self.make_blocks(128, num_blocks[1], stride=2)
        self.blocks3 = self.make_blocks(256, num_blocks[2], stride=2)
        self.blocks4 = self.make_blocks(512, num_blocks[3], stride=2)
        
        self.init_weights()

    def make_blocks(self, out_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                Bottleneck(self.in_planes, out_ch, stride)
            )
            self.in_planes = out_ch * self.expansion
        layers = nn.Sequential(*layers)
        return layers
    
    def init_weights(self):
        # MIT license
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
        # recommended init method
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if hasattr(m, 'zero_init_last_bn'):
                m.zero_init_last_bn()

    def forward(self, input, return_feats=False):
        x = self.stem(input)
        x = self.maxpool(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feats = self.feat_layer(x)
        out = self.cls_layer(feats)

        if return_feats:
            feats = nn.functional.normalize(feats, p=2.0, dim=1)
            return out, feats
        else:
            return out