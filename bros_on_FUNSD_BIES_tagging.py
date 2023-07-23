import datetime
import itertools
import json
import math
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from overrides import overrides
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
    EarlyStopping,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import itertools

from bros import BrosConfig, BrosTokenizer
from bros.modeling_bros import BrosForTokenClassification
from datasets import load_dataset, load_from_disk
from lightning_modules.schedulers import (
    cosine_scheduler,
    linear_scheduler,
    multistep_scheduler,
)

class FUNSDBIESDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_length=512,
        mode=None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        self.examples = load_dataset(self.dataset)[mode]
        self.bies_class_names = list(set(itertools.chain.from_iterable([set(example['labels']) for example in self.examples])))
        self.bies_class2idx = {label: idx for idx, label in enumerate(self.bies_class_names)}
        self.idx2bies_class = {idx: label for label, idx in self.bies_class2idx.items()}
        breakpoint()
        self.features = convert_examples_to_features(
            examples=self.examples,
            class2idx=self.bies_class2idx,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_label_id=-100,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feature = self.features[idx]
        example = self.examples[idx]

        width, height = feature.page_size
        bbox = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        cls_bb = np.array([0.0] * 8)

        actual_bboxes = np.array(feature.actual_bboxes, dtype=np.float32)
        actual_bboxes_len = len(feature.actual_bboxes)
        x1s = actual_bboxes[:, 0].reshape((actual_bboxes_len, 1)) / width
        y1s = actual_bboxes[:, 1].reshape((actual_bboxes_len, 1)) / height
        x2s = actual_bboxes[:, 2].reshape((actual_bboxes_len, 1)) / width
        y2s = actual_bboxes[:, 3].reshape((actual_bboxes_len, 1)) / height
        normalized_bboxes = torch.from_numpy(np.hstack((x1s, y1s, x2s, y1s, x2s, y2s, x1s, y2s)))

        bbox[:len(normalized_bboxes)] = normalized_bboxes
        bbox = torch.from_numpy(bbox)
        input_ids = torch.tensor(feature.input_ids)
        attention_mask = torch.tensor(feature.input_mask)
        labels = torch.tensor(feature.label_ids)

        return_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return return_dict


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        boxes,
        actual_bboxes,
        file_name,
        page_size,
    ):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size

def convert_examples_to_features(
    examples,
    class2idx,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token_segment_id=1,
    sep_token_extra=False,
    pad_on_left=False,
    cls_token_box=[0, 0, 0, 0],
    pad_token_box=[0, 0, 0, 0],
    pad_token_segment_id=0,
    pad_token_label_id=-1,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        file_name = example["file_name"]
        page_size = example["page_size"]
        width, height = page_size
        sep_token_box = [width, height, width, height]
        # if ex_index % 10000 == 0:
        #     print("Writing example {} of {}".format(ex_index, len(examples)))

        tokens = []
        token_boxes = []
        actual_bboxes = []
        label_ids = []
        for word, label, box, actual_bbox in zip(
            example["words"], example["labels"], example["boxes"], example["actual_bboxes"]
        ):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_bbox] * len(word_tokens))
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend(
                [class2idx[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
            )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [sep_token_box]
        label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_boxes = [cls_token_box] + token_boxes
        actual_bboxes = [cls_token_box] + actual_bboxes
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token_id] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        token_boxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
            )
        )
    return features


class BROSDataPLModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.train_batch_size = self.cfg.train.batch_size
        self.val_batch_size = self.cfg.val.batch_size
        self.train_dataset = None
        self.val_dataset = None

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        return loader


    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.cfg.val.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        return loader

    @overrides
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        return batch


class BROSModelPLModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = None

        self.optimizer_types = {
            "sgd": SGD,
            "adam": Adam,
            "adamw": AdamW,
        }
        self.loss_func = nn.CrossEntropyLoss()
        self.eval_kwargs = {
            "ignore_index": -1,
            "label_map": -1,
        }

    def training_step(self, batch, batch_idx, *args):
        _, loss = self.model_forward(batch)
        log_dict_input = {"train_loss": loss}
        self.log_dict(log_dict_input, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, *args):
        head_outputs, loss = self.model_forward(batch)
        step_out = do_eval_step(batch, head_outputs.logits, loss, self.eval_kwargs)
        return step_out

    def on_validation_epoch_end(self):
        ...
        # scores = do_eval_epoch_end(validation_step_outputs)
        # self.print(
        #     f"precision: {scores['precision']:.4f}, recall: {scores['recall']:.4f}, f1: {scores['f1']:.4f}"
        # )


    def model_forward(self, batch):
        # input_ids = batch["input_ids"]
        # bbox = batch["bbox"]
        # attention_mask = batch["attention_mask"]

        # head_outputs = self.model(
        #     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask
        # )

        # loss = self._get_loss(head_outputs.logits, batch)

        # return head_outputs, loss
        ...

    def _get_loss(self, head_outputs, batch):
        # # logits = head_outputs.view(-1, self.model_cfg.n_classes)
        # logits = head_outputs.view(-1, self.cfg.model.n_classes)
        # labels = batch["labels"].view(-1)

        # loss = self.loss_func(logits, labels)

        # return loss
        ...

    def setup(self, stage):
        self.time_tracker = time.time()

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)
        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def _get_lr_scheduler(self, optimizer):
        cfg_train = self.cfg.train
        lr_schedule_method = cfg_train.optimizer.lr_schedule.method
        lr_schedule_params = cfg_train.optimizer.lr_schedule.params

        if lr_schedule_method is None:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1)
        elif lr_schedule_method == "step":
            scheduler = multistep_scheduler(optimizer, **lr_schedule_params)
        elif lr_schedule_method == "cosine":
            total_samples = cfg_train.max_epochs * cfg_train.num_samples_per_epoch
            total_batch_size = cfg_train.batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = cosine_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        elif lr_schedule_method == "linear":
            total_samples = cfg_train.max_epochs * cfg_train.num_samples_per_epoch
            total_batch_size = cfg_train.batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = linear_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        else:
            raise ValueError(f"Unknown lr_schedule_method={lr_schedule_method}")

        return scheduler

    def _get_optimizer(self):
        opt_cfg = self.cfg.train.optimizer
        method = opt_cfg.method.lower()

        if method not in self.optimizer_types:
            raise ValueError(f"Unknown optimizer method={method}")

        kwargs = dict(opt_cfg.params)
        kwargs["params"] = self.model.parameters()
        optimizer = self.optimizer_types[method](**kwargs)

        return optimizer

    # @rank_zero_only
    # @overrides
    # def on_fit_end(self):
    #     hparam_dict = cfg_to_hparams(self.cfg, {})
    #     metric_dict = {"metric/dummy": 0}

    #     tb_logger = get_specific_pl_logger(self.logger, TensorBoardLogger)

    #     if tb_logger:
    #         tb_logger.log_hyperparams(hparam_dict, metric_dict)

    # @overrides
    # def training_epoch_end(self, training_step_outputs):
    #     avg_loss = torch.tensor(0.0).to(self.device)
    #     for step_out in training_step_outputs:
    #         avg_loss += step_out["loss"]

    #     log_dict = {"train_loss": avg_loss}
    #     self._log_shell(log_dict, prefix="train ")

    # def _log_shell(self, log_info, prefix=""):
    #     log_info_shell = {}
    #     for k, v in log_info.items():
    #         new_v = v
    #         if type(new_v) is torch.Tensor:
    #             new_v = new_v.item()
    #         log_info_shell[k] = new_v

    #     out_str = prefix.upper()
    #     if prefix.upper().strip() in ["TRAIN", "VAL"]:
    #         out_str += f"[epoch: {self.current_epoch}/{self.cfg.train.max_epochs}]"

    #     if self.training:
    #         lr = self.trainer._lightning_optimizers[0].param_groups[0]["lr"]
    #         log_info_shell["lr"] = lr

    #     for key, value in log_info_shell.items():
    #         out_str += f" || {key}: {round(value, 5)}"
    #     out_str += f" || time: {round(time.time() - self.time_tracker, 1)}"
    #     out_str += " secs."
    #     self.print(out_str, flush=True)
    #     self.time_tracker = time.time()




def train(cfg):
    cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints")
    cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs")
    cfg.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # pprint cfg
    print(OmegaConf.to_yaml(cfg))

    # set env
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock with tokenizer
    pl.seed_everything(cfg.seed)

    # Load Tokenizer (going to be used in dataset to to convert texts to input_ids)
    tokenizer = BrosTokenizer.from_pretrained(cfg.tokenizer_path)

    # Prepare FUNSD dataset
    train_dataset = FUNSDBIESDataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        mode='train'
    )

    val_dataset = FUNSDBIESDataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        mode='val'
    )

    # make data module & update data_module train and val dataset
    data_module = BROSDataPLModule(cfg)
    data_module.train_dataset = train_dataset
    data_module.val_dataset = val_dataset

    breakpoint()
    # Load BROS config & pretrained model
    ## update config
    bros_config = BrosConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
    bies_class_names = train_dataset.bies_class_names
    id2label = {idx: name for idx, name in enumerate(bies_class_names)}
    label2id = {name: idx for idx, name in id2label.items()}
    bros_config.id2label = id2label
    bros_config.label2id = label2id


if __name__ == "__main__":
    # load training config
    finetune_funsd_ee_bies_config = {
        "workspace": "./finetune_funsd_ee_bies",
        "exp_name": "bros-base-uncased_from_dict_config3",
        "tokenizer_path": "naver-clova-ocr/bros-base-uncased",
        "dataset": "jinho8345/bros-funsd-bies",
        "task": "ee",
        "seed": 1,
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
        "model": {
            "pretrained_model_name_or_path": "naver-clova-ocr/bros-base-uncased",
            "max_seq_length": 512,
        },
        "train": {
            "batch_size": 16,
            "num_samples_per_epoch": 526,
            "max_epochs": 30,
            "use_fp16": True,
            "accelerator": "gpu",
            "strategy": {"type": "ddp"},
            "clip_gradient_algorithm": "norm",
            "clip_gradient_value": 1.0,
            "num_workers": 0,
            "optimizer": {
                "method": "adamw",
                "params": {"lr": 5e-05},
                "lr_schedule": {"method": "linear", "params": {"warmup_steps": 0}},
            },
            "val_interval": 1,
        },
        "val": {"batch_size": 16, "num_workers": 0, "limit_val_batches": 1.0},
    }

    # convert dictionary to omegaconf and update config
    cfg = OmegaConf.create(finetune_funsd_ee_bies_config)
    train(cfg)
    # inference(cfg)
