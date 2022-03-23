import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, dim):
        super().__init__()

        expanded_channels = dim * self.expansion

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=dim,
                      bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(in_channels=dim, 
                      out_channels=expanded_channels,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=expanded_channels,
                      out_channels=dim,
                      kernel_size=1,
                      stride=1,
                      bias=False),
        )

        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        # if self.conv(x).shape[1:] != self.shortcut(x).shape[1:]:
        #     import pdb; pdb.set_trace() 
        out = self.conv(x) + self.shortcut(x)
        return out


class ConvNext(nn.Module):
    def __init__(self, block, block_nums, num_classes=7000, dropout=0) -> None:
        super().__init__()

        self.num_classes = num_classes

        dims=[96, 192, 384, 768]

        self.downsampling = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=dims[0],
                      kernel_size=4,
                      stride=4,
                      bias=False),
            nn.BatchNorm2d(dims[0]),
        )
        self.downsampling.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsampling.append(downsample_layer)

        stage1 = self._make_stage(dims[0], block, block_nums[0])
        stage2 = self._make_stage(dims[1], block, block_nums[1])
        stage3 = self._make_stage(dims[2], block, block_nums[2])
        stage4 = self._make_stage(dims[3], block, block_nums[3])
        self.stage = nn.ModuleList([stage1, stage2, stage3, stage4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.embedding = nn.Linear(dims[3], dims[3], bias=False)
        # if dropout == 0:
        #     self.classifier = nn.Sequential(
        #         nn.BatchNorm1d(dims[3]),
        #         nn.GELU(),
        #         nn.Linear(dims[3], self.num_classes)
        #     )
        # else:
        #     self.classifier = nn.Sequential(
        #         nn.BatchNorm1d(dims[3]),
        #         nn.GELU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(dims[3], self.num_classes)
        #     )
        if dropout == 0:
            self.classifier = nn.Linear(dims[3], 7000)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(dims[3], dims[3], bias=False),
                nn.BatchNorm1d(dims[3]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dims[3], self.num_classes)
            )
        

        self.apply(self._initialize_weights)

    def _make_stage(self, dim, block, block_num):
        stage = []
        for i in range(block_num):
            stage.append(block(dim))
        return nn.Sequential(*stage)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_feats=False):
        for i in range(4):
            x = self.downsampling[i](x)
            x = self.stage[i](x)

        avg_out = self.avgpool(x)
        feats = avg_out.reshape(avg_out.size(0), -1)
        # feats = self.embedding(feats)
        classifier_out = self.classifier(feats)
        if return_feats:
            feats = nn.functional.normalize(feats, p=2.0, dim=1)
            return classifier_out, feats
        else:
            return classifier_out


def convnext_t(dropout=0):
    return ConvNext(BottleNeck, [3,3,9,3], dropout=dropout)

def my_convnext(dropout=0, block_nums=[5,9,9,2]):
    return ConvNext(BottleNeck, block_nums, dropout=dropout)

if __name__ == '__main__':
    from torchsummary import summary

    device = 'cpu'
    model = my_convnext().to(device)

    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print(f"Number of Params: {num_trainable_parameters}") 
    print(f"Less than 35m? {num_trainable_parameters<35000000}")
    
    summary(model, (3, 224, 224), device=device)