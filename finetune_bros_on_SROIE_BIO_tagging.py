import itertools
import json
import os
import numpy as np
import time
from overrides import overrides
from omegaconf import OmegaConf

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

from lightning_modules.bros_bio_module import do_eval_step, eval_ee_bio_batch, eval_ee_bio_example, parse_from_seq, do_eval_epoch_end

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR

from lightning_modules.schedulers import (
    cosine_scheduler,
    linear_scheduler,
    multistep_scheduler,
)
from utils import cfg_to_hparams, get_specific_pl_logger



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
            # worker_init_fn=self.seed_worker,
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
# class DonutModelPLModule(pl.LightningModule):
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
        self.eval_kwargs = None


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
        # batch["are_box_first_tokens"] : [1, 512]
        mask = batch["are_box_first_tokens"].view(-1) # [512]

        # head_outputs : torch.Size([1, 512, 9])
        logits = head_outputs.view(-1, self.cfg.model.n_classes) # [512, 9]
        logits = logits[mask] # torch.Size([105, 9])

        labels = batch["bio_labels"].view(-1) # torch.Size([512])
        labels = labels[mask] # torch.Size([105])

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




# Load Config
config_path = "configs/finetune_sroie_ee_bio_custom.yaml"
cfg = OmegaConf.load(config_path)
print(cfg)

# Load Tokenizer
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


# set env
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock with tokenizer
seed_everything(cfg.seed)
# pl.utilities.seed.seed_everything(cfg.get("seed", 1), workers=True)

# make data module
data_module = BROSDataPLModule(cfg)
# add datasets to data module
data_module.train_dataset = train_dataset
data_module.val_dataset = val_dataset

# Load BROS model
bros_config = BrosConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)

bros_config.num_labels = len(train_dataset.bio_class_names) # 4 classes * 2 (Beginning, Inside) + 1 (Outside)
cfg.model.n_classes = bros_config.num_labels

bros_model = BrosForTokenClassification.from_pretrained(
    cfg.model.pretrained_model_name_or_path,
    config=bros_config
)
model_module = BROSModelPLModule(cfg)
model_module.model = bros_model
model_module.eval_kwargs = {"class_names": train_dataset.class_names}

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
