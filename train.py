import argparse

from transformers import T5Model, T5Tokenizer

from .data import get_loader


def main():
    tokenizer = T5Tokenizer('t5-small')


if __name__ == '__main__':
    main()
