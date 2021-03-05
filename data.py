from functools import partial
from typing import List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from googletrans import Translator


class TranslationDataset(Dataset):
    def __init__(self, sentences_ru, sentences_en):
        self.sentences_ru = sentences_ru
        self.sentences_en = sentences_en
        self.translator = Translator()

    def __getitem__(self, index):
        src = self.sentences_ru[index]
        target = self.sentences_en[index]
        return src, target

    def augment_ru(self, text):
        return cycle_augmentation(text, self.translator, src='ru', dest='en')

    def augment_en(self, text):
        return cycle_augmentation(text, self.translator, src='en', dest='ru')

    def __len__(self):
        return len(self.sentences_en)


def cycle_augmentation(text, translator, src, dest):
    translated_text = translator.translate(text, src=src, dest=dest).text
    retranslated_text = translator.translate(translated_text, src=dest, dest=src).text
    return retranslated_text


def collate_fn_wo_tokenizer(batch, tokenizer):
    src = [sample[0] for sample in batch]
    target = [sample[1] for sample in batch]

    src_encoded = tokenizer(src, padding=True, return_tensors="pt").input_ids
    target_encoded = tokenizer(target, padding=True, return_tensors="pt").input_ids

    return src_encoded, target_encoded, src


def get_loader(data_src: List[str], data_target: List[str], tokenizer, **kwargs):
    collate_fn = partial(collate_fn_wo_tokenizer, tokenizer=tokenizer)

    dataset = TranslationDataset(data_src, data_target)
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        **kwargs,
    )
    return loader

def load_yandex_data(en_path, ru_path):
    en_text = Path(en_path).read_text().split('\n')
    ru_text = Path(ru_path).read_text().split('\n')

    return en_text, ru_text

def indices_to_words(tokenizer, indices):
    return tokenizer.decode(indices)
