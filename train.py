from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5Config, AdamW, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer

from data import get_loader, load_yandex_data


def main():
    root = Path('/home/vladbakhteev/data/mt')
    ru_path = root / 'corpus.en_ru.1m.ru'
    en_path = root / 'corpus.en_ru.1m.en'

    train_loader, valid_loader, tokenizer = get_loaders_and_tokenizer(en_path, ru_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = T5Config.from_pretrained('t5-small')
    model = T5ForConditionalGeneration(config=config)
    model.to(device)
    optim = AdamW(model.parameters(), lr=5e-5)

    epochs = 1
    for epoch in range(epochs):
        model.train()
        it = tqdm(train_loader, total=len(train_loader))
        for i, (src_batch, target_batch, _) in enumerate(it):
            src_batch, target_batch = src_batch.to(device), target_batch.to(device)

            optim.zero_grad()
            loss = model(input_ids=src_batch, labels=target_batch).loss
            loss.backward()
            optim.step()

            it.set_description(f'{loss.item():.3f}')

        model.eval()
        predicted, gt = [], []
        it = tqdm(valid_loader, total=len(valid_loader))
        for src_batch, _, src_strings in it:
            src_batch = src_batch.to(device)

            with torch.no_grad():
                output = model.generate(src_batch).cpu()
            output = [tokenizer.decode(o) for o in output]

            gt += output
            predicted += src_strings

        score = bleu_score(predicted, gt)
        print(f'Epoch {epoch}, BLEU: {score}')


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
