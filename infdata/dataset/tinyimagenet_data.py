"""
Download Tiny ImageNet Dataset
"""

import torch
from torchvision import datasets
import os
curr_dir = os.path.dirname(__file__)

import glob
import zipfile
import pandas as pd
import numpy as np
import cv2

from infdata.transformation.tinyimagenet_tf import AlbumTransforms
from torch.utils.data import Dataset

def download_data():
    import wget

    data_folder = os.path.join(curr_dir, "../../", "data")

    if not os.path.exists(os.path.join(data_folder, 'tiny-imagenet-200/')):
        print('Downloading dataset...')

        # The URL for the dataset zip file.
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_file = os.path.join(data_folder, "tiny-imagenet-200.zip")

        # Download the file (if we haven't already)
        if not os.path.exists(zip_file):
            wget.download(url, zip_file)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

        os.remove(zip_file)
        return True
    else:
        return False

class CreateData(object):

    def __init__(self, data_folder):
        self.data_folder = data_folder

    def _create_classes(self):
        self.classes = pd.read_table(os.path.join(self.data_folder,"wnids.txt"), names=['class'])["class"].values
        self.classes_df = pd.read_table(os.path.join(self.data_folder,"words.txt"), names=["class","class_name"])
        self.classes_df = self.classes_df[self.classes_df["class"].isin(self.classes)].reset_index(drop=True)

        self.classtoidx = {y:i for i,y in enumerate(self.classes_df["class"].unique())}
        def addidx(x):
            return self.classtoidx[x]

        self.classes_df["target"] = self.classes_df["class"].apply(addidx)

        self.orgclasstoidx = dict(zip(self.classes_df["class_name"],self.classes_df["target"]))


    def _read_val_dataframe(self):
        val_annotations = pd.read_table(os.path.join(self.data_folder,"val/val_annotations.txt"), names=["image_name","class","x1","x2","x3","x4"])
        self.val_df = val_annotations[val_annotations["class"].isin(self.classes)].reset_index(drop=True)

    def _read_train_dataframe(self):
        self.train_df = pd.DataFrame(columns=['image_name',"class","x1","x2","x3","x4"])

        for name in glob.glob(os.path.join(self.data_folder,"train")+"/*"):
            class_name = name.rsplit("/")[-1]
            table_name = os.path.join(name,f"{class_name}_boxes.txt")
            annots = pd.read_table(table_name, names=["image_name","x1","x2","x3","x4"])
            annots["class"] = class_name
            annots = annots[['image_name',"class","x1","x2","x3","x4"]]

            self.train_df = self.train_df.append(annots, ignore_index=True)

    def _create_complete_dataframe(self):
        self.dataset = self.train_df.append(self.val_df, ignore_index=True).sample(frac=1).reset_index(drop=True)
        self.dataset["target"] = self.dataset["class"].apply(lambda x : self.classtoidx[x])

    def _create_train_test_split(self):
        from sklearn.model_selection import train_test_split
        self.train_df, self.val_df = train_test_split(self.dataset, test_size=0.3, stratify=self.dataset['target'])

    def create_dataset(self):
        boolean = download_data()
        print()
        if boolean:
            print("Dataset Downloaded Successfully")
        else:
            print("Data is already downloaded..")

        self._create_classes()
        self._read_val_dataframe()
        self._read_train_dataframe()
        self._create_complete_dataframe()
        self._create_train_test_split()

        self.classes_df.to_csv(os.path.join(self.data_folder, "classes.csv"), index=False)
        self.train_df.to_csv(os.path.join(self.data_folder,"train.csv"), index=False)
        self.val_df.to_csv(os.path.join(self.data_folder,"val.csv"), index=False)

class TinyImageNetDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform):
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(self.root_dir, csv_file))
        print(f"Dataframe Size : {self.df.shape[0]}")
        self.target = torch.tensor(np.asarray(self.df["target"].values))
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]["image_name"]
        fname = image_name.split("_")[0]
        
        if fname=="val":
            image_path = os.path.join(self.root_dir,"val","images",image_name)
        else:
            image_path = os.path.join(self.root_dir, "train", fname, "images", image_name)

        im = cv2.imread(image_path)
            
        if self.transform:
            im = self.transform(im)

        target = self.target[idx]
        
        return im, target


class DownloadData(object):
    def __init__(self):
        self.data_path = os.path.join(curr_dir,"../../","data/tiny-imagenet-200")
        createdata = CreateData(self.data_path)
        createdata.create_dataset()

        transf = AlbumTransforms()
        self.tinyimagenet_traindata = TinyImageNetDataset("train.csv", self.data_path,
                                                transform=transf.get_train_transforms())
        
        self.tinyimagenet_validdata = TinyImageNetDataset("val.csv", self.data_path,
                                                transform=transf.get_valid_transforms())

        self.classes2idx = createdata.orgclasstoidx




