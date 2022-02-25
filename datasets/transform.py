import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']

train_transforms = A.Compose(
    [
        A.RandomResizedCrop(224, 224, scale=(0.7, 1), ratio=(0.9, 1.1)), 
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.CenterCrop(height=200, width=200),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)