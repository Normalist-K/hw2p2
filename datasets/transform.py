import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']

image_net_norm_mean = (0.485, 0.456, 0.406)
image_net_norm_std = (0.229, 0.224, 0.225)
face_norm_mean = (0.511, 0.402, 0.351)
face_norm_std = (0.270, 0.235, 0.222)

train_transforms = A.Compose(
    [
        # A.Resize(224, 224), 
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.1, 
                           rotate_limit=15, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=face_norm_mean, 
                std=face_norm_std, 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()
    ]
)

val_transforms = A.Compose(
    [
        # A.CenterCrop(height=200, width=200),
        A.Resize(224, 224),
        A.Normalize(
                mean=face_norm_mean, 
                std=face_norm_std, 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2(),
    ]
)