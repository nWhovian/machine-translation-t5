from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from transformers import T5Config, AdamW, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer
from collections import OrderedDict

from data import get_loader, load_yandex_data


class T5Model(pl.LightningModule):
    def __init__(self):
        super(T5Model, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = T5Config.from_pretrained('t5-small')
        model = T5ForConditionalGeneration(config=config)
        model.to(device)
        self.model = model

        root = Path('1mcorpus')
        ru_path = root / 'corpus.en_ru.1m.ru'
        en_path = root / 'corpus.en_ru.1m.en'
        self.train_loader, self.valid_loader, self.tokenizer = get_loaders_and_tokenizer(en_path, ru_path)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=5e-5,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src_batch = batch[0]
        target_batch = batch[1]
        src_batch = src_batch.to(device)
        target_batch = target_batch.to(device)

        loss = self.model(
            input_ids=src_batch,
            labels=target_batch,
        ).loss

        output = OrderedDict({
            "loss": loss,
        })
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return output

    def validation_step(self, batch, batch_idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src_batch = batch[0]
        src_strings = batch[2]
        src_batch = src_batch.to(device)

        with torch.no_grad():
            output = self.model.generate(src_batch).cpu()
        output = [self.tokenizer.decode(o) for o in output]

        output = OrderedDict({
            "gt": output,
            "predicted": src_strings,
        })
        return output

    def validation_end(self, outputs):
        gt = [out['gt'] for out in outputs]
        predicted = [out['predicted'] for out in outputs]
        bleu_score = pl.metrics.nlp.BLEUScore(predicted, gt)

        tqdm_dict = {
            "bleu_score": bleu_score,
        }
        result = {
                  "bleu_score": bleu_score,
                  }
        self.log('bleu_score', bleu_score, on_step=False, on_epoch=True, prog_bar=True)
        return result

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


def main():
    early_stop_callback = EarlyStopping(
        monitor="bleu_score",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="min"
    )

    trainer = pl.Trainer(
        gpus=None,
        # checkpoint_callback=checkpoint_callback,
        # max_epochs=100,
        # early_stop_callback=early_stop_callback,
    )

    model = T5Model()
    trainer.fit(model)
    trainer.test()


def get_loaders_and_tokenizer(en_path, ru_path):
    en_text, ru_text = load_yandex_data(en_path, ru_path)
    en_train, en_valid, ru_train, ru_valid = train_test_split(en_text, ru_text, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    train_loader = get_loader(
        data_src=ru_train,
        data_target=en_train,
        tokenizer=tokenizer,
        batch_size=16,
        shuffle=True,
        drop_last=True
    )
    valid_loader = get_loader(
        data_src=ru_valid,
        data_target=en_valid,
        tokenizer=tokenizer,
        batch_size=2,
        shuffle=False,
        drop_last=False
    )

    return train_loader, valid_loader, tokenizer


if __name__ == '__main__':
    main()
