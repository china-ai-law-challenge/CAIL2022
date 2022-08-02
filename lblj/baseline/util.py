from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Metric, load_dataset, load_metric
from transformers import (BatchEncoding, EvalPrediction, PreTrainedModel,
                          Trainer, TrainingArguments)
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTrainedTokenizerBase)


def preprocess_function(
    examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List[str]]:
    sc_sentences: List[str] = sum([
        [f'诉方称：{sc}'] * 5 for sc in examples['sc']
    ], [])

    bc_sentences: List[str] = sum([
        [f'辩方回应：{examples[f"bc_{j}"][i]}' for j in range(1, 6)]
        for i in range(len(examples['id']))
    ], [])

    tokenized_examples: BatchEncoding = tokenizer(
        sc_sentences, bc_sentences, truncation=True, max_length=512
    )

    return {k: [v[i:i + 5] for i in range(0, len(v), 5)]
            for k, v in tokenized_examples.items()}


def get_dataset(
    location: str, tokenizer: PreTrainedTokenizerBase
) -> Union[Dataset, DatasetDict]:
    dataset: Dataset = load_dataset('json', data_files=location, split='train')

    dataset = dataset.remove_columns(
        ['text_id', 'category', 'chapter', 'crime']
    ).map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

    if 'answer' in dataset.column_names:
        return dataset.train_test_split(test_size=0.1)
    else:
        return dataset


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])

        if 'answer' in features[0].keys():
            labels: Optional[List[int]] = [
                feature.pop('answer') for feature in features
            ]
        else:
            labels = None

        flattened_features: List[Dict[str, Any]] = sum([
            [{k: v[i] for k, v in feature.items()}
             for i in range(num_choices)] for feature in features
        ], [])

        padded_features: BatchEncoding = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        batch: Dict[str, torch.Tensor] = {
            k: v.view(batch_size, num_choices, -1)
            for k, v in padded_features.items()
        }

        if labels is not None:
            batch['labels'] = torch.tensor(labels, dtype=torch.int64) - 1
        return batch


def compute_metrics(
    eval_pred: EvalPrediction, metric: Metric
) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def get_trainer(
    train_set: Optional[Dataset], test_set: Optional[Dataset],
    tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel
) -> Trainer:
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        weight_decay=0.01,
        no_cuda=not torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        optim='adamw_torch',
        report_to='none'
    )

    metric = load_metric('accuracy')

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=lambda x: compute_metrics(x, metric)
    )
