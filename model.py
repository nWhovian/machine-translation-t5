import math

import pytorch_lightning as pl
from transformers import T5Config, AdamW, T5ForConditionalGeneration
from torchtext.data.metrics import bleu_score


class T5Model(pl.LightningModule):
    def __init__(self, tokenizer):
        super(T5Model, self).__init__()
        self.tokenizer = tokenizer
        config = T5Config.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration(config=config)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=1e-2)

    def training_step(self, batch, batch_idx):
        src_batch, target_batch, _ = batch

        loss = self.model(
            input_ids=src_batch,
            labels=target_batch,
        ).loss

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        src_batch, _, src_strings = batch

        output = self.model.generate(src_batch)
        output = [self.tokenizer.decode(o) for o in output]
        return {"gt": src_strings, "predicted": output}

    def validation_epoch_end(self, outputs):
        gt, predicted = [], []
        for out in outputs:
            gt += out['gt']
            predicted += out['predicted']

        # score = bleu_score(predicted, gt)
        # self.log('bleu_score', score, on_step=False, on_epoch=True, prog_bar=True)

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure):
        adjust_optim(optimizer, batch_nb)

        optimizer.step()
        # Faster than optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None


def adjust_optim(optimizer, n_iter, k=10000):
    optimizer.param_groups[0]['lr'] = 1 / math.sqrt(max(n_iter, k))
