from functools import partial
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(self, sentences_ru, sentences_en):
        self.sentences_ru = sentences_ru
        self.sentences_en = sentences_en

    def __getitem__(self, index):
        src = self.sentences_ru[index]
        target = self.sentences_en[index]
        return src, target

    def __len__(self):
        return len(self.sentences_en)


def collate_fn_wo_tokenizer(batch, tokenizer):
    src = [sample[0] for sample in batch]
    target = [sample[1] for sample in batch]

    src_encoded = tokenizer(src, padding=True).input_ids
    target_encoded = tokenizer(target, padding=True).input_ids

    src_encoded = torch.tensor(src_encoded)
    target_encoded = torch.tensor(target_encoded)

    return src_encoded, target_encoded


def get_loader(data_src: List[str], data_target: List[str], tokenizer, **kwargs):
    collate_fn = partial(collate_fn_wo_tokenizer, tokenizer=tokenizer)

    dataset = TranslationDataset(data_src, data_target)
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        **kwargs,
    )
    return loader
