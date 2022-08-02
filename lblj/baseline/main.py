from json import dumps
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import torch
from jieba import lcut
from datasets import Dataset
from transformers import (BertForMultipleChoice, BertTokenizer, Trainer,
                          set_seed)

from util import get_dataset, get_trainer


def match(target: str, candidates: List[str]) -> Tuple[int, float]:
    target_words: Set[str] = set(lcut(target))

    overlap: torch.Tensor = torch.tensor(
        [len(target_words.intersection(lcut(x))) for x in candidates],
        dtype=torch.float
    ).log_softmax(0)

    prediction: int = torch.argmax(overlap).item()
    return prediction + 1, overlap[prediction]


def matching_predict(dataset: Dataset) -> List[Tuple[int, float]]:
    return [match(x['sc'], [x[f'bc_{i}'] for i in range(1, 6)])
            for x in dataset]


def bert_predict(logits: np.ndarray) -> List[Tuple[int, float]]:
    log_softmax: torch.Tensor = torch.tensor(logits).log_softmax(1)
    predictions: List[Tuple[int, float]] = []

    for x in log_softmax:
        prediction: int = torch.argmax(x).item()
        predictions.append((prediction + 1, x[prediction].item()))

    return predictions


if __name__ == '__main__':
    set_seed(42)
    model_path = 'model'

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
    model: BertForMultipleChoice = BertForMultipleChoice.from_pretrained(
        model_path
    )

    test_set: Dataset = get_dataset('data/test_entry.jsonl', tokenizer)
    trainer: Trainer = get_trainer(None, None, tokenizer, model)

    matching_pred: List[Tuple[int, float]] = matching_predict(test_set)
    bert_pred: List[Tuple[int, float]] = bert_predict(
        trainer.predict(test_set).predictions
    )

    Path('output').mkdir(exist_ok=True)

    with open('output/predict.jsonl', 'w', encoding='utf8') as f:
        for i, id in enumerate(test_set['id']):
            print(dumps({
                'id': id,
                'answer': bert_pred[i][0]
                if bert_pred[i][1] > matching_pred[i][1]
                else matching_pred[i][0]
            }), file=f)
