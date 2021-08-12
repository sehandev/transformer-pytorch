# Standard

# PIP
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Custom
from config import Config
from c_dataset import CustomDataModule
from module import CustomModule


tb_logger = TensorBoardLogger('logs/', name='train')

cfg = Config()

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,  # 성능 향상을 평가하는 최소 변화량
    patience=10,  # wait at least n epoch
    verbose=True,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkout/',
    filename='transformer-{epoch:03d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min',
)

trainer = Trainer(
    gpus=cfg.NUM_GPUS,
    max_epochs=cfg.MAX_EPOCHS,
    logger=tb_logger,  # tensorboard logger
    progress_bar_refresh_rate=1,  # log every n step(batch)
    accelerator="ddp",  # DDP for multi-gpu
    deterministic=True,  # for reproducibility
    precision=16,  # use Mixed Precision for speed up
    callbacks=[
        early_stop_callback,
        checkpoint_callback,
    ],
)

data_module = CustomDataModule(
    batch_size=cfg.BATCH_SIZE,
    num_workers=cfg.NUM_WORKERS,
)

model = CustomModule(
    seq_len=cfg.SEQ_LEN,
    learning_rate=cfg.LEARNING_RATE,
    criterion_name=cfg.CRITERION,
    optimizer_name=cfg.OPTIMIZER,
)

print('Start model fitting')
trainer.fit(model, data_module)

print('Start testing')
trainer.test(datamodule=data_module)

print(f'Best model : {checkpoint_callback.best_model_path}')
