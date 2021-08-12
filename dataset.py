# Standard

# PIP
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchtext.datasets import WikiText2

# Custom


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=1,
        num_workers=0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(
        self,
        stage=None,
    ):
        # Assign train & val datasets
        if stage == "fit" or stage is None:
            self.train_dataset = WikiText2(split='train')
            self.valid_dataset = WikiText2(split='valid')

        # Assign test dataset
        if stage == "test" or stage is None:
            self.test_dataset = WikiText2(split='test')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
