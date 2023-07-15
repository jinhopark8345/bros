import itertools
import json
import os
import numpy as np
import time
from overrides import overrides
from omegaconf import OmegaConf
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from datasets import load_dataset
from datasets import load_from_disk
from pytorch_lightning import Trainer
from transformers import AutoTokenizer

from utils import get_class_names

import math
import random
import re
from pathlib import Path

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.seed import seed_everything
from torch.optim.lr_scheduler import LambdaLR

from utils import get_callbacks, get_config, get_loggers, get_plugins
from bros.modeling_bros import BrosForTokenClassification
from bros import BrosTokenizer
from bros import BrosConfig

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR

from lightning_modules.schedulers import (
    cosine_scheduler,
    linear_scheduler,
    multistep_scheduler,
)
from utils import cfg_to_hparams, get_specific_pl_logger
from seqeval.metrics import f1_score, precision_score, recall_score

class FUNSDBIODataset(Dataset):
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
        self.bies_class_names = list(set(chain.from_iterable([set(example['labels']) for example in self.examples])))
        self.bies_class2idx = {label: idx for idx, label in enumerate(self.bies_class_names)}
        self.idx2bies_class = {idx: label for label, idx in self.bies_class2idx.items()}
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

    def model_forward(self, batch):
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]

        head_outputs = self.model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask
        )

        loss = self._get_loss(head_outputs.logits, batch)

        return head_outputs, loss

    def _get_loss(self, head_outputs, batch):
        # logits = head_outputs.view(-1, self.model_cfg.n_classes)
        logits = head_outputs.view(-1, self.cfg.model.n_classes)
        labels = batch["labels"].view(-1)

        loss = self.loss_func(logits, labels)

        return loss

    @overrides
    def training_step(self, batch, batch_idx, *args):
        _, loss = self.model_forward(batch)
        log_dict_input = {"train_loss": loss}
        self.log_dict(log_dict_input, sync_dist=True)
        return loss

    @torch.no_grad()
    @overrides
    def validation_step(self, batch, batch_idx, *args):
        head_outputs, loss = self.model_forward(batch)
        step_out = do_eval_step(batch, head_outputs.logits, loss, self.eval_kwargs)
        return step_out

    @torch.no_grad()
    @overrides
    def validation_epoch_end(self, validation_step_outputs):
        scores = do_eval_epoch_end(validation_step_outputs)
        self.print(
            f"precision: {scores['precision']:.4f}, recall: {scores['recall']:.4f}, f1: {scores['f1']:.4f}"
        )

    @torch.no_grad()
    @overrides
    def validation_epoch_end(self, validation_step_outputs):
        scores = do_eval_epoch_end(validation_step_outputs)
        self.print(
            f"precision: {scores['precision']:.4f}, recall: {scores['recall']:.4f}, f1: {scores['f1']:.4f}"
        )

    @overrides
    def setup(self, stage):
        self.time_tracker = time.time()

    @overrides
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

    @rank_zero_only
    @overrides
    def on_fit_end(self):
        hparam_dict = cfg_to_hparams(self.cfg, {})
        metric_dict = {"metric/dummy": 0}

        tb_logger = get_specific_pl_logger(self.logger, TensorBoardLogger)

        if tb_logger:
            tb_logger.log_hyperparams(hparam_dict, metric_dict)

    @overrides
    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.tensor(0.0).to(self.device)
        for step_out in training_step_outputs:
            avg_loss += step_out["loss"]

        log_dict = {"train_loss": avg_loss}
        self._log_shell(log_dict, prefix="train ")

    def _log_shell(self, log_info, prefix=""):
        log_info_shell = {}
        for k, v in log_info.items():
            new_v = v
            if type(new_v) is torch.Tensor:
                new_v = new_v.item()
            log_info_shell[k] = new_v

        out_str = prefix.upper()
        if prefix.upper().strip() in ["TRAIN", "VAL"]:
            out_str += f"[epoch: {self.current_epoch}/{self.cfg.train.max_epochs}]"

        if self.training:
            lr = self.trainer._lightning_optimizers[0].param_groups[0]["lr"]
            log_info_shell["lr"] = lr

        for key, value in log_info_shell.items():
            out_str += f" || {key}: {round(value, 5)}"
        out_str += f" || time: {round(time.time() - self.time_tracker, 1)}"
        out_str += " secs."
        self.print(out_str, flush=True)
        self.time_tracker = time.time()

def do_eval_step(batch, head_outputs, loss, eval_kwargs):
    ignore_index = eval_kwargs["ignore_index"]
    label_map = eval_kwargs["label_map"]

    pr_labels = torch.argmax(head_outputs, -1).cpu().numpy()

    labels = batch["labels"]
    gt_labels = labels.cpu().numpy()

    prs, gts = [], []
    # https://github.com/microsoft/unilm/blob/master/layoutlm/deprecated/examples/seq_labeling/run_seq_labeling.py#L372
    bsz, max_seq_length = labels.shape
    for example_idx in range(bsz):
        example_prs, example_gts = [], []
        for token_idx in range(max_seq_length):
            if labels[example_idx, token_idx] != ignore_index:
                example_prs.append(label_map[pr_labels[example_idx, token_idx]])
                example_gts.append(label_map[gt_labels[example_idx, token_idx]])
        prs.append(example_prs)
        gts.append(example_gts)

    step_out = {
        "loss": loss,
        "prs": prs,
        "gts": gts,
    }

    return step_out


def do_eval_epoch_end(step_outputs):
    prs, gts = [], []
    for step_out in step_outputs:
        prs.extend(step_out["prs"])
        gts.extend(step_out["gts"])

    precision = precision_score(gts, prs)
    recall = recall_score(gts, prs)
    f1 = f1_score(gts, prs)

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return scores



# Load Config
config_path = "configs/finetune_funsd_ee_bies_custom.yaml"
cfg = OmegaConf.load(config_path)
print(cfg)

# Load Tokenizer
tokenizer = BrosTokenizer.from_pretrained(cfg.tokenizer_path)

train_dataset = FUNSDBIODataset(
    dataset=cfg.dataset,
    tokenizer=tokenizer,
    max_seq_length=cfg.model.max_seq_length,
    mode='train'
)

val_dataset = FUNSDBIODataset(
    dataset=cfg.dataset,
    tokenizer=tokenizer,
    max_seq_length=cfg.model.max_seq_length,
    mode='val'
)

# set env
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock with tokenizer
seed_everything(cfg.seed)

# make data module
data_module = BROSDataPLModule(cfg)
# add datasets to data module
data_module.train_dataset = train_dataset
data_module.val_dataset = val_dataset

# Load BROS model
bros_config = BrosConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)

bros_config.num_labels = len(train_dataset.bies_class_names) # 4 classes * 2 (Beginning, Inside) + 1 (Outside)
cfg.model.n_classes = bros_config.num_labels

bros_model = BrosForTokenClassification.from_pretrained(
    cfg.model.pretrained_model_name_or_path,
    config=bros_config
)
breakpoint()
model_module = BROSModelPLModule(cfg)
model_module.model = bros_model
model_module.eval_kwargs = {
    "label_map": train_dataset.idx2bies_class,
    "ignore_index": -100
}

cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints")
cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs")


callbacks = get_callbacks(cfg)
plugins = get_plugins(cfg)
loggers = get_loggers(cfg)

trainer = Trainer(
    accelerator=cfg.train.accelerator,
    gpus=torch.cuda.device_count(),
    max_epochs=cfg.train.max_epochs,
    gradient_clip_val=cfg.train.clip_gradient_value,
    gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
    callbacks=callbacks,
    plugins=plugins,
    sync_batchnorm=True,
    precision=16 if cfg.train.use_fp16 else 32,
    terminate_on_nan=False,
    replace_sampler_ddp=False,
    move_metrics_to_cpu=False,
    progress_bar_refresh_rate=0,
    check_val_every_n_epoch=cfg.train.val_interval,
    logger=loggers,
    benchmark=cfg.cudnn_benchmark,
    deterministic=cfg.cudnn_deterministic,
    limit_val_batches=cfg.val.limit_val_batches,
)

trainer.fit(model_module, data_module)
