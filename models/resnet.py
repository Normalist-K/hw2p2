from re import M
import torch
import torch.nn as nn
import math

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        out = self.act(out)
        return out


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        expanded_channels = out_channels * self.expansion

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=expanded_channels,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(expanded_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=expanded_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(expanded_channels)
            )
        
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, block_nums, num_classes=7000) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1)
        )

        self.in_channels = 64
        self.stage1 = self._make_stage(64, block, block_nums[0], 1)
        self.stage2 = self._make_stage(128, block, block_nums[1], 2)
        self.stage3 = self._make_stage(256, block, block_nums[2], 2)
        self.stage4 = self._make_stage(512, block, block_nums[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.in_channels, self.num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.in_channels, 1200, bias=False),
        #     nn.BatchNorm1d(1200),
        #     nn.GELU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(1200, num_classes)
        # )

    def _make_stage(self, out_channels, block, block_num, stride):
        stage = []
        for i in range(block_num):
            if i >= 1:
                stride = 1
            stage.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*stage)

    def forward(self, x, return_feats=False):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        avg_out = self.avgpool(out)
        feats = avg_out.reshape(avg_out.size(0), -1)
        classifier_out = self.classifier(feats)
        if return_feats:
            feats = nn.functional.normalize(feats, p=2.0, dim=1)
            return classifier_out, feats
        else:
            return classifier_out

def resnet34():
    return ResNet(BasicBlock, [3, 3, 9, 3])

def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def myresnet():
    return ResNet(BottleNeck, [4, 4, 6, 2])