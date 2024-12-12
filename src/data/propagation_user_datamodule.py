from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import pickle as pkl
import torch.nn.functional as F
from transformers import BertTokenizer
import pandas as pd
import os
import re
import fnmatch
import json

class DiffusionDataset(Dataset):
    def __init__(self, data_path,num_classes):
        self.data_path = data_path
        #self.transform = transform
        self.num_classes = num_classes
        self.samples = self.load_samples()

    def load_samples(self):

        data_df = pd.read_csv(self.data_path)
        # text_list = list(data_df["text"])
        # label_list = list(data_df["label"])
        
        seq = list(data_df["network_id"])
        seq = [json.loads(item) for item in seq]
        len_seq = list(data_df["network_len"])
        target = list(data_df["target"])
        id = list(data_df["id"])
        # image_path_list = []
        text_list = list(data_df["text"])
    
        combined_list = list(zip(seq,len_seq,target,id,text_list))

        return combined_list
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq,len_seq,target,id,text = self.samples[idx]
        # data preprocessing
        # target = torch.tensor(int(target))
        # final_target = F.one_hot(target,num_classes=self.num_classes).float()
        # if self.tokenizer is not None:
        #     text = self.tokenizer(text,return_tensors='pt',truncation = True,padding='max_length',max_length=512)
        #print(text.shape)
        # seq = torch.LongTensor(seq)
        # len_seq = torch.LongTensor(len_seq)
        # target = torch.LongTensor(int(target))
        # seq = [[item] for item in seq]
        return seq,len_seq,target,id,text
    
    def find_image_path(self,id,origin_path):
        """
            find image path
        """
        files_list = os.listdir(origin_path)
        pattern = f"{id}.*"
        matching_files = fnmatch.filter(files_list,pattern)
        
        if matching_files:
            image_path = os.path.join(origin_path,matching_files[0])
            return image_path
        else:
            return None
        
class DiffusionDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir : str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        tokenizer: str = "",
        num_classes : int = 2,
    ):
        super().__init__()
        #self.data_dir = data_dir
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    @property
    def num_classes(self):
        return 2
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        dataset = DiffusionDataset(data_path = self.hparams.data_dir,num_classes=self.hparams.num_classes)
        # if stage=="fit" or stage is None:
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=dataset,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(45),
        )
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
if __name__ == "__main__":
    test = SnopesDataModule()