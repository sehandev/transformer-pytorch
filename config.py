# Standard
from os.path import dirname, abspath, join

# PIP
import pytorch_lightning as pl

# Custom


class Config:
    def __init__(self):
        # for train
        self.SEED = 42
        self.NUM_GPUS = 1
        self.MAX_EPOCHS = 100
        self.BATCH_SIZE = 128
        self.CRITERION = 'RMSE'
        self.OPTIMIZER = 'AdamW'
        self.LEARNING_RATE = 1e-6
        self.NUM_WORKERS = 4
        self.VERBOSE = 0  # 0: quiet, 1: with log
        self.DATA_DIR = 'data'

        # Hyperparamters
        self.SEQ_LEN = 256
        self.NUM_VOCABS = 30000
        self.MAX_LEN = 512
        self.D_MODEL = 256
        self.EMBEDDING_DROPOUT = 0.1

        self.project_dir = dirname(abspath(__file__))
        self.DATA_DIR = join(self.project_dir, self.DATA_DIR)

        self.set_random_seed()

    def set_random_seed(self):
        pl.seed_everything(self.SEED)

    def update_hparams(
        self,
        seed=None,
        num_gpus=None,
        batch_size=None,
        learning_rate=None,
    ):
        if seed:
            self.SEED = seed
            self.set_random_seed(seed)
        self.NUM_GPUS = num_gpus if num_gpus else self.NUM_GPUS
        self.BATCH_SIZE = batch_size if batch_size else self.BATCH_SIZE
        self.LEARNING_RATE = learning_rate if learning_rate else self.LEARNING_RATE

    def __str__(self):
        report = '[ Config ]\n'
        report += f'SEED: {self.SEED}\n'
        report += f'NUM_GPUS: {self.NUM_GPUS}\n'
        report += f'MAX_EPOCHS: {self.MAX_EPOCHS}\n'
        report += f'BATCH_SIZE: {self.BATCH_SIZE}\n'
        report += f'CRITERION: {self.CRITERION}\n'
        report += f'OPTIMIZER: {self.OPTIMIZER}\n'
        report += f'LEARNING_RATE: {self.LEARNING_RATE}\n'
        report += f'NUM_WORKERS: {self.NUM_WORKERS}\n'
        report += f'VERBOSE: {self.VERBOSE}\n'
        report += f'DATA_DIR: {self.DATA_DIR}\n'

        return report


if __name__ == '__main__':
    cfg = Config()

    print(cfg)

    cfg.update_hparams(
        seed=10,
        learning_rate=1e-4,
    )

    print(cfg)
