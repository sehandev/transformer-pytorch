# Standard

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

# Custom
from c_model import CustomModel


class CustomModule(LightningModule):
    def __init__(
        self,
        seq_len,
        learning_rate,
        criterion_name='RMSE',
        optimizer_name='Adam',
        momentum=0.9,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.model = CustomModel()

        self.criterion = self.get_loss_function(criterion_name)
        self.optimizer = self.get_optimizer(optimizer_name)

    @staticmethod
    def get_loss_function(loss_function_name):
        name = loss_function_name.lower()

        if name == 'CrossEntropy'.lower():
            return nn.CrossEntropyLoss()

        raise ValueError(f'{loss_function_name} is not on the custom criterion list!')

    def get_optimizer(self, optimizer_name):
        name = optimizer_name.lower()

        if name == 'SGD'.lower():
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if name == 'Adam'.lower():
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if name == 'AdamW'.lower():
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        raise ValueError(f'{optimizer_name} is not on the custom optimizer list!')

    def forward(self, x):

        out = self.model(x)
        # out : (batch_size, window_size, d_model)

        return out

    def configure_optimizers(self):
        return self.optimizer

    def common_step(self, x):
        if not state in ['train', 'valid', 'test']:
            raise(f'{state} must be one of train, valid, test')

        y_hat = self(x)
        # y_hat : (batch_size, sequence_length, d_model)

        loss = 0
        for batch_y_hat, batch_x in zip(y_hat, x):
            batch_loss = self.criterion(batch_y_hat, batch_x)
            loss += batch_loss

        loss /= len(x)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x : (batch_size, sequence_length)

        loss = self.common_step(x)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x : (batch_size, sequence_length)

        loss = self.common_step(x)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # x : (batch_size, sequence_length)

        loss = self.common_step(x)

        self.log('test_loss', loss, sync_dist=True)
        return loss
