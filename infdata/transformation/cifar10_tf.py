from torchvision import transforms
import numpy as np

from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    Resize,
    Cutout,
    Rotate,
    RandomResizedCrop,
    RandomBrightnessContrast
)

from albumentations.pytorch import ToTensor

class Transforms(object):
    def __init__(self):
        self.train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
		    transforms.RandomHorizontalFlip(),
		    transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(0.25)
        ])

        self.valid_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

class AlbumTransforms(object):
    def __init__(self):
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2023, 0.1994, 0.2010])

        self.transforms_elist = [ 
            RandomResizedCrop(height=32, width=32),
            HorizontalFlip(p=0.5),
            # RandomBrightnessContrast(),
			# Rotate(limit=10),
			Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=self.mean*255.0, p=0.25),
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







