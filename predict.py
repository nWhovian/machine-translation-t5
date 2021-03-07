import torch
from transformers import AutoTokenizer

from model import T5Model


def main():
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5Model.load_from_checkpoint('../epoch=1-step=199999.ckpt', tokenizer=tokenizer).model
    model.eval()

    src_str = "повар, какова твоя профессия?"

    input_ids = tokenizer(src_str, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids)

    translation = tokenizer.decode(output[0])

    print(translation)


if __name__ == '__main__':
    main()
