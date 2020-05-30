from torchvision import transforms
import cv2
from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    Resize,
    CoarseDropout,
    Rotate,
    GaussianBlur,
    HueSaturationValue
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
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

        self.transforms_elist = [ 
            HueSaturationValue(p=0.25),
			HorizontalFlip(p=0.5),
			Rotate(limit=15),
			CoarseDropout(max_holes=1, max_height=16, max_width=16, min_height=4,
						    min_width=4, fill_value=mean*255.0, p=0.75)
        ]

        self.transforms_test = [
            Resize(32,32)
        ]

        self.transforms_main = [
            Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
		    ToTensor()
        ]


    def get_train_transforms(self):
        train_transforms = Compose(self.transforms_elist.extend(self.transforms_main))
        return lambda img:train_transforms(image=np.array(img))["image"]

    def get_valid_transforms(self):
        valid_transforms = Compose(self.transforms_main)
        return lambda img:valid_transforms(image=np.array(img))["image"]

    def get_test_transforms(self):
        test_transforms = Compose(self.transforms_test.extend(self.transforms_main))
        return lambda img:test_transforms(image=np.array(img))["image"]







