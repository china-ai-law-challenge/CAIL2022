from datasets import DatasetDict
from transformers import (BertForMultipleChoice, BertTokenizer, Trainer,
                          set_seed)

from util import get_dataset, get_trainer

if __name__ == '__main__':
    set_seed(42)
    model_card: str = 'bert-base-chinese'

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_card)
    model: BertForMultipleChoice = BertForMultipleChoice.from_pretrained(
        model_card
    )

    split_set: DatasetDict = get_dataset('data/train_entry.jsonl', tokenizer)
    trainer: Trainer = get_trainer(split_set['train'], split_set['test'],
                                   tokenizer, model)

    trainer.train()
    tokenizer.save_pretrained('model')
    model.save_pretrained('model')
