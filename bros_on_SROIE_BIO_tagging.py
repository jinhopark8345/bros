import itertools
import json
import os
import numpy as np
import time
import yaml
import datetime
from overrides import overrides
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pprint import pprint

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer

import math
import random
import re
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR

from lightning_modules.schedulers import (
    cosine_scheduler,
    linear_scheduler,
    multistep_scheduler,
)

from bros.modeling_bros import BrosForTokenClassification
from bros import BrosTokenizer
from bros import BrosConfig


class SROIEBIODataset(Dataset):
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
        self.class_names = list(self.examples['parse'][0]['class'].keys())
        self.class_idx_dic = dict(
            [(class_name, idx) for idx, class_name in enumerate(self.class_names)]
        )

        self.bio_class_names = ["O"]
        for class_name in self.class_names:
            self.bio_class_names.extend([f"B_{class_name}", f"I_{class_name}"])
        self.bio_class_idx_dic = dict(
            [
                (bio_class_name, idx)
                for idx, bio_class_name in enumerate(self.bio_class_names)
            ]
        )


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        json_obj = self.examples[idx]

        width = json_obj["meta"]["imageSize"]["width"]
        height = json_obj["meta"]["imageSize"]["height"]

        input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        bbox = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)

        bio_labels = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)

        list_tokens = []
        list_bbs = []
        box2token_span_map = []

        cls_bbs = [0.0] * 8

        for word_idx, word in enumerate(json_obj["words"]):
            tokens = word["tokens"]
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                break

            box2token_span_map.append(
                [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
            )  # including st_idx
            list_tokens += tokens

            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

            bb = list(itertools.chain(*bb))
            bbs = [bb for _ in range(len(tokens))]
            list_bbs.extend(bbs)

        sep_bbs = [width, height] * 4

        # For [CLS] and [SEP]
        list_tokens = (
            [self.cls_token_id]
            + list_tokens[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(list_bbs) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            list_bbs = [cls_bbs] + list_bbs[: self.max_seq_length - 2] + [sep_bbs]

        len_list_tokens = len(list_tokens)
        input_ids[:len_list_tokens] = list_tokens
        attention_mask[:len_list_tokens] = 1

        bbox[:len_list_tokens, :] = list_bbs

        # Normalize bbox -> 0 ~ 1
        bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
        bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height

        # Label
        classes_dic = json_obj["parse"]["class"]
        for class_name in self.class_names:
            if class_name == "O":
                continue
            if class_name not in classes_dic:
                continue
            if classes_dic[class_name] is None:
                continue

            for word_list in classes_dic[class_name]:
                # At first, connect the class and the first box
                is_first, last_word_idx = True, -1
                for word_idx in word_list:
                    if word_idx >= len(box2token_span_map):
                        break
                    box2token_span_start, box2token_span_end = box2token_span_map[
                        word_idx
                    ]
                    for converted_word_idx in range(
                        box2token_span_start, box2token_span_end
                    ):
                        if converted_word_idx >= self.max_seq_length:
                            break

                        if is_first:
                            bio_labels[converted_word_idx] = self.bio_class_idx_dic[
                                f"B_{class_name}"
                            ]
                            is_first = False
                        else:
                            bio_labels[converted_word_idx] = self.bio_class_idx_dic[
                                f"I_{class_name}"
                            ]

            st_indices, _ = zip(*box2token_span_map)
            st_indices = [
                st_idx for st_idx in st_indices if st_idx < self.max_seq_length
            ]
            are_box_first_tokens[st_indices] = True

        input_ids = torch.from_numpy(input_ids)
        bbox = torch.from_numpy(bbox)
        attention_mask = torch.from_numpy(attention_mask)

        bio_labels = torch.from_numpy(bio_labels)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)

        return_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "bio_labels": bio_labels,
            "are_box_first_tokens": are_box_first_tokens,
        }

        return return_dict

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

def eval_ee_bio_example(pr_seq, gt_seq, box_first_token_mask, class_names):
    valid_gt_seq = gt_seq[box_first_token_mask]
    valid_pr_seq = pr_seq[box_first_token_mask]

    gt_parse = parse_from_seq(valid_gt_seq, class_names)
    pr_parse = parse_from_seq(valid_pr_seq, class_names)

    n_gt_classes, n_pr_classes, n_correct_classes = 0, 0, 0
    for class_idx in range(len(class_names)):
        # Evaluate by ID
        n_gt_classes += len(gt_parse[class_idx])
        n_pr_classes += len(pr_parse[class_idx])
        n_correct_classes += len(gt_parse[class_idx] & pr_parse[class_idx])

    return n_gt_classes, n_pr_classes, n_correct_classes


def parse_from_seq(seq, class_names):
    parsed = [[] for _ in range(len(class_names))]
    for i, label_id_tensor in enumerate(seq):
        label_id = label_id_tensor.item()

        if label_id == 0:  # O
            continue

        class_id = (label_id - 1) // 2
        is_b_tag = label_id % 2 == 1

        if is_b_tag:
            parsed[class_id].append((i,))
        elif len(parsed[class_id]) != 0:
            parsed[class_id][-1] = parsed[class_id][-1] + (i,)

    parsed = [set(indices_list) for indices_list in parsed]

    return parsed

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
        self.class_names = None
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx, *args):

        # unpack batch
        input_ids = batch['input_ids']
        bbox = batch['bbox']
        attention_mask = batch['attention_mask']
        box_first_token_mask = batch['are_box_first_tokens']
        labels = batch['bio_labels']

        # inference model
        output = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            box_first_token_mask=box_first_token_mask,
            labels=labels
        )

        # log result
        self.log_dict({"train_loss": output.loss}, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx, *args):

        # unpack batch
        input_ids = batch['input_ids']
        bbox = batch['bbox']
        attention_mask = batch['attention_mask']
        are_box_first_tokens = batch['are_box_first_tokens']
        gt_labels = batch['bio_labels']
        labels = batch['bio_labels']

        # inference model
        output = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            box_first_token_mask=are_box_first_tokens,
            labels=labels
        )

        pred_labels = torch.argmax(output.logits, -1)


        n_batch_gt_classes, n_batch_pred_classes, n_batch_correct_classes = 0, 0, 0

        batch_size = output.logits.shape[0]
        for example_idx in range(batch_size):
            n_gt_classes, n_pred_classes, n_correct_classes = eval_ee_bio_example(
                pred_labels[example_idx],
                gt_labels[example_idx],
                are_box_first_tokens[example_idx],
                self.class_names,
            )

            n_batch_gt_classes += n_gt_classes
            n_batch_pred_classes += n_pred_classes
            n_batch_correct_classes += n_correct_classes

        step_out = {
            "loss": output.loss,
            "n_batch_gt_classes": n_batch_gt_classes,
            "n_batch_pr_classes": n_batch_pred_classes,
            "n_batch_correct_classes": n_batch_correct_classes,
        }

        return step_out

    def validation_epoch_end(self, validation_step_outputs):
        n_total_gt_classes, n_total_pr_classes, n_total_correct_classes = 0, 0, 0

        for step_out in validation_step_outputs:
            n_total_gt_classes += step_out["n_batch_gt_classes"]
            n_total_pr_classes += step_out["n_batch_pr_classes"]
            n_total_correct_classes += step_out["n_batch_correct_classes"]

        precision = 0.0 if n_total_pr_classes == 0 else n_total_correct_classes / n_total_pr_classes
        recall = 0.0 if n_total_gt_classes == 0 else n_total_correct_classes / n_total_gt_classes
        f1 = 0.0 if recall * precision == 0 else 2.0 * recall * precision / (recall + precision)

        self.log_dict({
                "precision": precision,
                "recall": recall,
                "f1": f1,
        })

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

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.cfg.get('exp_name', '')}"
        items["exp_version"] = f"{self.cfg.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.cfg.workspace) / self.cfg.exp_name / self.cfg.exp_version
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


# load training config
finetune_sroie_ee_bio_config = {
    "workspace": "./finetune_sroie_ee_bio",
    "exp_name": "bros-base-uncased_from_dict_config3",
    "tokenizer_path": "naver-clova-ocr/bros-base-uncased",
    "dataset": "jinho8345/bros-sroie",
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
        "num_workers": 4,
        "optimizer": {
            "method": "adamw",
            "params": {"lr": 5e-05},
            "lr_schedule": {"method": "linear", "params": {"warmup_steps": 0}},
        },
        "val_interval": 1,
    },
    "val": {"batch_size": 8, "num_workers": 4, "limit_val_batches": 1.0},
}

# convert dictionary to omegaconf and update config
cfg = OmegaConf.create(finetune_sroie_ee_bio_config)
cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints")
cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs")
cfg.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# pprint config
pprint(cfg)

# set env
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock with tokenizer
pl.seed_everything(cfg.seed)

# Load Tokenizer (going to be used in dataset to to convert texts to input_ids)
tokenizer = BrosTokenizer.from_pretrained(cfg.tokenizer_path)

# Load SROIE dataset
train_dataset = SROIEBIODataset(
    dataset=cfg.dataset,
    tokenizer=tokenizer,
    max_seq_length=cfg.model.max_seq_length,
    mode='train'
)

val_dataset = SROIEBIODataset(
    dataset=cfg.dataset,
    tokenizer=tokenizer,
    max_seq_length=cfg.model.max_seq_length,
    mode='val'
)

# make data module & update data_module train and val dataset
data_module = BROSDataPLModule(cfg)
data_module.train_dataset = train_dataset
data_module.val_dataset = val_dataset

# Load BROS config & pretrained model
bros_config = BrosConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)

## update model config
bros_config.num_labels = len(train_dataset.bio_class_names) # 4 classes * 2 (Beginning, Inside) + 1 (Outside)

## update training config
cfg.model.n_classes = bros_config.num_labels

## load pretrained model
bros_model = BrosForTokenClassification.from_pretrained(
    cfg.model.pretrained_model_name_or_path,
    config=bros_config
)

# model module setting
model_module = BROSModelPLModule(cfg)
model_module.model = bros_model
model_module.class_names = train_dataset.class_names


# define trainer callbacks, plugins, loggers
# callbacks = get_callbacks(cfg)
# plugins = get_plugins(cfg)
# loggers = get_loggers(cfg)

loggers = TensorBoardLogger(
    save_dir=cfg.workspace,
    name=cfg.exp_name,
    version=cfg.exp_version,
    default_hp_metric=False,
)
lr_callback = LearningRateMonitor(logging_interval="step")
checkpoint_callback = ModelCheckpoint(
    monitor="f1",
    dirpath=Path(cfg.workspace) / cfg.exp_name / cfg.exp_version,
    filename="artifacts",
    save_top_k=1,
    save_last=False,
    mode="min",
)

# define Trainer
trainer = pl.Trainer(
    resume_from_checkpoint=cfg.get("resume_from_checkpoint_path", None),
    num_nodes=cfg.get("num_nodes", 1),
    gpus=torch.cuda.device_count(),
    strategy="ddp",
    accelerator=cfg.train.accelerator,

    max_epochs=cfg.train.max_epochs,
    gradient_clip_val=cfg.train.clip_gradient_value,
    gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,

    # fast_dev_run=True,
    num_sanity_val_steps=3,
    # plugins=plugins,

    precision=16 if cfg.train.use_fp16 else 32,
    logger=loggers,
    callbacks=[lr_callback, checkpoint_callback],
)

# train
trainer.fit(model_module, data_module)

# finetuned_model = BrosForTokenClassification(bros_config)

# def load_model_weight(model, pretrained_model_file):
#     pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")[
#         "state_dict"
#     ]
#     new_state_dict = {}
#     for k, v in pretrained_model_state_dict.items():
#         new_k = k
#         if new_k.startswith("net."):
#             new_k = new_k[len("net.") :]
#         new_state_dict[new_k] = v
#     model.load_state_dict(new_state_dict)


# load_model_weight(finetuned_model, cfg.pretrained_model_file)

# # load fine-tuned model

# # evalute
# step_outputs = []
# for example_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
#     # Convert batch tensors to given device
#     for k in batch.keys():
#         if isinstance(batch[k], torch.Tensor):
#             batch[k] = batch[k].to(finetuned_model.backbone.device)

#     with torch.no_grad():
#         head_outputs, loss = finetuned_model(batch)
#     step_out = do_eval_step(batch, head_outputs, loss, eval_kwargs)
#     step_outputs.append(step_out)

# # Get scores
# scores = do_eval_epoch_end(step_outputs)
# print(
#     f"precision: {scores['precision']:.4f}, "
#     f"recall: {scores['recall']:.4f}, "
#     f"f1: {scores['f1']:.4f}"
# )
