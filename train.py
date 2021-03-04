from tqdm import tqdm
from transformers import T5Config, AdamW, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer

from data import get_loader, load_yandex_data, indices_to_words


def main():
    ru_path = '1mcorpus/corpus.en_ru.1m.ru'
    en_path = '1mcorpus/corpus.en_ru.1m.en'
    en_text, ru_text = load_yandex_data(en_path, ru_path)
    en_train, en_valid, ru_train, ru_valid = train_test_split(en_text, ru_text, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    train_loader = get_loader(
        data_src=ru_train,
        data_target=en_train,
        tokenizer=tokenizer,
        batch_size=2,
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

    config = T5Config.from_pretrained('t5-small')
    model = T5ForConditionalGeneration(config=config)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    print('training')
    epochs = 1
    for epoch in range(epochs):
        it = tqdm(train_loader, total=len(train_loader))
        for src_batch, target_batch in it:
            optim.zero_grad()
            loss = model(input_ids=src_batch, labels=target_batch).loss
            loss.backward()
            optim.step()
            it.set_description(str(loss.item()))

    model.eval()

    print('validation')
    predicted, true = [], []
    it = tqdm(valid_loader, total=len(valid_loader))
    for src_batch, target_batch in it:
        output = model(input_ids=src_batch)
        # predicted.extend(indices_to_words(tokenizer, output))
        # true.extend(indices_to_words(tokenizer, target_batch))

    score = bleu_score(predicted, true)
    print('the BLEU score: ', score)


if __name__ == '__main__':
    main()
