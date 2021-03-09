import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from model import T5Model


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('weights', type=str)
    parser.add_argument('--sent', type=str, required=False, default='', help='Sentence to translate')
    parser.add_argument('--file', type=str, required=False, default='', help='File with sentences to translate')
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5Model.load_from_checkpoint(args.weights, tokenizer=tokenizer).model
    model.eval()

    sentences = []
    if args.sent:
        sentences += [args.sent]
    if args.file:
        sentences += Path(args.file).read_text().split('\n')

    translations = []
    for src in tqdm(sentences):
        translations.append(
            translate_sentence(src, tokenizer, model)[6:].replace('</s>', '')
        )

    output = '\n'.join(translations)
    Path('answer.txt').write_text(output)


def translate_sentence(src: str, tokenizer, model):
    input_ids = tokenizer(src, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids)
    translation = tokenizer.decode(output[0])
    return translation


if __name__ == '__main__':
    main()
