from datasets import load_dataset, load_from_disk

### Load FUNSD dataset

# dataset = load_dataset("nielsr/funsd")
dataset = load_dataset("nielsr/funsd-iob-original")
# dataset = load_from_disk("/home/jinho/Projects/bros/save/dataset")

label_list = dataset["train"].features["ner_tags"].feature.names
id2label = {id: label for id, label in enumerate(label_list)}
print(f"{id2label = } = ")


print(f'{dataset["train"].features} = ')

### Load Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("/home/jinho/Projects/bros/save/tokenizer")

### Add quad feature
# The first thing we'll do is add a "quads" column to the dataset, which contains the quadratics per word (8 numbers).
def add_quad(batch):
    images = batch["image"]
    boxes = batch["original_bboxes"]

    quads = []
    for image, boxes_example in zip(images, boxes):
        width, height = image.size
        quads_example = []
        for box in boxes_example:
            quad = [
                box[0] / width,
                box[1] / height,
                box[2] / width,
                box[1] / height,
                box[2] / width,
                box[3] / height,
                box[0] / width,
                box[3] / height,
            ]
            quads_example.append(quad)

        quads.append(quads_example)

    batch["quads"] = quads

    return batch


dataset = dataset.map(add_quad, batched=True)

example = dataset["train"][0]
print(example["words"])
print(example["quads"])


def create_new_examples(batch):
    examples = {}
    inputs = tokenizer(
        batch["words"],
        is_split_into_words=True,
        truncation=True,
        return_overflowing_tokens=True,
        max_length=100,
    )

    input_ids = []
    bbox = []
    labels = []
    for batch_idx in range(len(inputs.input_ids)):
        input_ids_example = inputs.input_ids[batch_idx]
        word_ids_example = inputs.word_ids(batch_index=batch_idx)
        org_batch_index = inputs["overflow_to_sample_mapping"][batch_idx]

        bbox_example = []
        labels_example = []
        for id, word_id in zip(input_ids_example, word_ids_example):
            if id == 101:
                bbox_example.append([0.0] * 8)
                labels_example.append(-100)
            elif id == 102:
                bbox_example.append([1.0] * 8)
                labels_example.append(-100)
            else:
                bbox_example.append(batch["quads"][org_batch_index][word_id])
                labels_example.append(batch["ner_tags"][org_batch_index][word_id])

        input_ids.append(input_ids_example)
        bbox.append(bbox_example)
        labels.append(labels_example)

    encoding = {}
    encoding["input_ids"] = input_ids
    encoding["bbox"] = bbox
    encoding["labels"] = labels

    return encoding


encoded_dataset = dataset.map(
    create_new_examples, batched=True, remove_columns=dataset["train"].column_names
)

example = encoded_dataset["train"][1]
print(example.keys())
print(tokenizer.decode(example["input_ids"]))

for id, label in zip(example["input_ids"], example["labels"]):
    if label != -100:
        print(tokenizer.decode([id]), id2label[label])
    else:
        print(tokenizer.decode([id]), label)


import torch

### Define PyTorch DataLoader
from torch.utils.data import DataLoader


def collate_fn(features):
    boxes = [feature["bbox"] for feature in features]
    labels = [feature["labels"] for feature in features]
    # use tokenizer to pad input_ids
    batch = tokenizer.pad(features, padding="max_length", max_length=512)

    sequence_length = torch.tensor(batch["input_ids"]).shape[1]
    batch["labels"] = [
        labels_example + [-100] * (sequence_length - len(labels_example))
        for labels_example in labels
    ]
    batch["bbox"] = [
        boxes_example
        + [[0, 0, 0, 0, 0, 0, 0, 0]] * (sequence_length - len(boxes_example))
        for boxes_example in boxes
    ]

    # convert to PyTorch
    # batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
    batch = {k: torch.tensor(v) for k, v in batch.items()}

    return batch

train_dataloader = DataLoader(
    encoded_dataset["train"], batch_size=2, shuffle=True, collate_fn=collate_fn
)
eval_dataloader = DataLoader(
    encoded_dataset["test"], batch_size=2, shuffle=False, collate_fn=collate_fn
)

### Define Model

from bros import BrosForTokenClassification

model = BrosForTokenClassification.from_pretrained("naver-clova-ocr/bros-base-uncased", id2label=id2label)
# model = BrosForTokenClassification.from_pretrained("/home/jinho/Projects/bros/save/model", id2label=id2label)

import evaluate

metric = evaluate.load("seqeval")

import numpy as np
# from seqeval.metrics import classification_report

return_entity_level_metrics = False


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test",
                                  num_train_epochs=100,
                                  learning_rate=5e-5,
                                  evaluation_strategy="steps",
                                  eval_steps=100,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1")

from transformers.data.data_collator import default_data_collator

class CustomTrainer(Trainer):
  def get_train_dataloader(self):
    return train_dataloader

  def get_eval_dataloader(self, eval_dataset = None):
    return eval_dataloader

# Initialize our Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
