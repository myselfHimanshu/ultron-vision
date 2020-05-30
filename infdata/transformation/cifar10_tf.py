from torchvision import transforms

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
            trasforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])