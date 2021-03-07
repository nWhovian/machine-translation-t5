from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from data import get_loader, load_yandex_data
from model import T5Model


def main():
    root = Path('/home/vbv/data/mt')
    ru_path = root / 'corpus.en_ru.1m.ru'
    en_path = root / 'corpus.en_ru.1m.en'

    train_loader, valid_loader, tokenizer = get_loaders_and_tokenizer(en_path, ru_path)
    model = T5Model(tokenizer)

    checkpoint_callback = ModelCheckpoint()
    logger = TensorBoardLogger('tb_logs', name='machine-translation-logs')
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=5,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
    trainer.test()


def get_loaders_and_tokenizer(en_path, ru_path, smaller_size=None):
    en_text, ru_text = load_yandex_data(en_path, ru_path)
    if smaller_size is not None:
        en_text = en_text[:smaller_size]
        ru_text = ru_text[:smaller_size]

    en_train, en_valid, ru_train, ru_valid = train_test_split(en_text, ru_text, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    train_loader = get_loader(
        data_src=ru_train,
        data_target=en_train,
        tokenizer=tokenizer,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )
    valid_loader = get_loader(
        data_src=ru_valid,
        data_target=en_valid,
        tokenizer=tokenizer,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=1,
    )

    return train_loader, valid_loader, tokenizer


if __name__ == '__main__':
    main()
