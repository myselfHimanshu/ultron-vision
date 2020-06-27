import numpy as np

from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    Resize,
    PadIfNeeded,
    RandomCrop,
    Cutout,
    Rotate,
    RandomResizedCrop,
    RandomBrightnessContrast
)

from albumentations.pytorch import ToTensor

class AlbumTransforms(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.transforms_elist = [ 
            PadIfNeeded(min_height=72, min_width=72, value=self.mean*255.0), 
            RandomCrop(height=64, width=64, p=1.0),
            HorizontalFlip(p=0.5),
            Rotate(7, p=0.5),
            Cutout(num_holes=2, max_h_size=8, max_w_size=8, fill_value=self.mean*255.0, p=0.5),
        ]

        self.transforms_test = [
            Resize(32,32),
        ]

        self.transforms_main = [
            Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
		    ToTensor(),
        ]


    def get_train_transforms(self):
        train_transforms = Compose(self.transforms_elist + self.transforms_main)
        return lambda img:train_transforms(image=np.array(img))["image"]

    def get_valid_transforms(self):
        valid_transforms = Compose(self.transforms_main)
        return lambda img:valid_transforms(image=np.array(img))["image"]

    def get_test_transforms(self):
        test_transforms = Compose(self.transforms_test + self.transforms_main)
        return lambda img:test_transforms(image=np.array(img))["image"]