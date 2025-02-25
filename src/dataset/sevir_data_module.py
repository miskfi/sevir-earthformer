import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from .sevir_dataset import SEVIRTorchDataset


class SEVIRIndividualDataModule(pl.LightningDataModule):
    """Data module providing the train, validation and test dataloaders"""

    def __init__(self, train_p=0.7, val_p=0.2, batch_size=16, num_workers=0, **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_p = train_p
        self.val_p = val_p

        self.dataset = SEVIRTorchDataset(**dataset_kwargs)

        train_idx, val_idx, test_idx = self.get_split_idx()

        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.test_dataset = Subset(self.dataset, test_idx)

    def get_split_idx(self):
        total = len(self.dataset)
        idxs = list(range(total))
        train_size = int(self.train_p * total)
        val_size = int(self.val_p * total)

        return idxs[:train_size], idxs[train_size : (train_size + val_size)], idxs[(train_size + val_size) :]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    @property
    def num_train_samples(self):
        return len(self.train_dataset)
