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

from bros import BrosConfig, BrosTokenizer
from bros.modeling_bros import BrosForTokenClassification
from datasets import load_dataset, load_from_disk
from lightning_modules.schedulers import (
    cosine_scheduler,
    linear_scheduler,
    multistep_scheduler,
)


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
        self.class_names = list(self.examples["parse"][0]["class"].keys())
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
        are_box_end_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)

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

            st_indices, end_indices = zip(*box2token_span_map)
            st_indices = [
                st_idx for st_idx in st_indices if st_idx < self.max_seq_length
            ]
            end_indices = [
                end_idx for end_idx in end_indices if end_idx < self.max_seq_length
            ]

            assert len(st_indices) == len(end_indices)

            are_box_first_tokens[st_indices] = True
            are_box_end_tokens[end_indices] = True

        input_ids = torch.from_numpy(input_ids)
        bbox = torch.from_numpy(bbox)
        attention_mask = torch.from_numpy(attention_mask)

        bio_labels = torch.from_numpy(bio_labels)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
        are_box_end_tokens = torch.from_numpy(are_box_end_tokens)

        return_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "bio_labels": bio_labels,
            "are_box_first_tokens": are_box_first_tokens,
            "are_box_end_tokens": are_box_end_tokens,
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
    def __init__(self, cfg, tokenizer):
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
        self.bio_class_names = None
        self.tokenizer = tokenizer
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx, *args):
        # unpack batch
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]
        box_first_token_mask = batch["are_box_first_tokens"]
        labels = batch["bio_labels"]

        # inference model
        prediction = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            box_first_token_mask=box_first_token_mask,
            labels=labels,
        )

        loss = prediction.loss
        self.log_dict({"train_loss": loss}, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, *args):
        # unpack batch
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]
        are_box_first_tokens = batch["are_box_first_tokens"]
        are_box_end_tokens  = batch["are_box_end_tokens"]
        gt_labels = batch["bio_labels"]
        labels = batch["bio_labels"]


        # inference model
        prediction = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            box_first_token_mask=are_box_first_tokens,
            labels=labels,
        )

        val_loss = predictin.loss
        pred_labels = torch.argmax(prediction.logits, -1)

        n_batch_gt_classes, n_batch_pred_classes, n_batch_correct_classes = 0, 0, 0
        batch_size = prediction.logits.shape[0]

        for example_idx, (pred_label, gt_label, box_first_token_mask, box_end_token_mask) in enumerate(
            zip(pred_labels, gt_labels, are_box_first_tokens, are_box_end_tokens)
        ):


            # validation loss : # calculate validation loss of "box_first_tokens" only
            valid_gt_label = gt_label[box_first_token_mask]
            valid_pred_label = pred_label[box_first_token_mask]

            gt_parse = parse_from_seq(valid_gt_label, self.class_names)
            pred_parse = parse_from_seq(valid_pred_label, self.class_names)

            """
            (Pdb++) valid_gt_label
            tensor([3, 4, 4, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')

            --> after parse

            (Pdb++) gt_parse
            [{(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)}, {(0, 1, 2, 3)}, {(45,)}, {(113,)}]
            """


            n_gt_classes = sum([len(gt_parse[class_idx]) for class_idx in range(len(self.class_names))])
            n_pred_classes = sum([len(pred_parse[class_idx]) for class_idx in range(len(self.class_names))])
            n_correct_classes = sum(
                [len(gt_parse[class_idx] & pred_parse[class_idx]) for class_idx in range(len(self.class_names))]
            )
            n_batch_gt_classes += n_gt_classes
            n_batch_pred_classes += n_pred_classes
            n_batch_correct_classes += n_correct_classes



            box_first_token_idx2ori_idx = box_first_token_mask.nonzero(as_tuple=True)[0]
            box2token_span_maps = torch.hstack((
                (box_first_token_mask == True).nonzero(),
                (box_end_token_mask == True).nonzero()
            )).cpu().numpy()
            start_token_idx2end_token_idx = {e[0]:e[1] for e in box2token_span_maps}

            pred_cls2text = {name: [] for name in self.class_names}
            gt_cls2text = deepcopy(pred_cls2text)
            correct_cls2text = deepcopy(pred_cls2text)
            incorrect_cls2text = deepcopy(pred_cls2text)
            for cls_idx, cls_name in enumerate(self.class_names):
                # all pred text for cls
                for box_first_token_indices in pred_parse[cls_idx]:
                    ori_indices = box_first_token_idx2ori_idx[torch.tensor(box_first_token_indices)].cpu().tolist()
                    text_span = torch.tensor(list(range(ori_indices[0], start_token_idx2end_token_idx[ori_indices[-1]])))
                    pred_text = self.tokenizer.decode(input_ids[example_idx][text_span])
                    pred_cls2text[cls_name].append(pred_text)

                # all gt text for cls
                for box_first_token_indices in gt_parse[cls_idx]:
                    ori_indices = box_first_token_idx2ori_idx[torch.tensor(box_first_token_indices)].cpu().tolist()
                    text_span = torch.tensor(list(range(ori_indices[0], start_token_idx2end_token_idx[ori_indices[-1]])))
                    gt_text = self.tokenizer.decode(input_ids[example_idx][text_span])
                    gt_cls2text[cls_name].append(gt_text)

                # all correct text for cls
                for box_first_token_indices in pred_parse[cls_idx] & gt_parse[cls_idx]:
                    ori_indices = box_first_token_idx2ori_idx[torch.tensor(box_first_token_indices)].cpu().tolist()
                    text_span = torch.tensor(list(range(ori_indices[0], start_token_idx2end_token_idx[ori_indices[-1]])))
                    correct_text = self.tokenizer.decode(input_ids[example_idx][text_span])
                    correct_cls2text[cls_name].append(correct_text)

                # all incorrect text for cls (text in gt but not in pred + text not in gt but in pred)
                for box_first_token_indices in pred_parse[cls_idx] ^ gt_parse[cls_idx]:
                    ori_indices = box_first_token_idx2ori_idx[torch.tensor(box_first_token_indices)].cpu().tolist()
                    text_span = torch.tensor(list(range(ori_indices[0], start_token_idx2end_token_idx[ori_indices[-1]])))
                    incorrect_text = self.tokenizer.decode(input_ids[example_idx][text_span])
                    incorrect_cls2text[cls_name].append(incorrect_text)

        print(f"{pred_cls2text = }")
        print(f"{gt_cls2text = }")
        print(f"{correct_cls2text = }")
        print(f"{incorrect_cls2text = }")

        step_out = {
            "n_batch_gt_classes": n_batch_gt_classes,
            "n_batch_pr_classes": n_batch_pred_classes,
            "n_batch_correct_classes": n_batch_correct_classes,
        }

        self.validation_step_outputs.append(step_out)
        self.log_dict({"val_loss": val_loss}, sync_dist=True, prog_bar=True)
        self.log_dict(step_out, sync_dist=True)

        return step_out

    def on_validation_epoch_end(self):
        all_preds = self.validation_step_outputs

        n_total_gt_classes, n_total_pr_classes, n_total_correct_classes = 0, 0, 0

        for step_out in all_preds:
            n_total_gt_classes += step_out["n_batch_gt_classes"]
            n_total_pr_classes += step_out["n_batch_pr_classes"]
            n_total_correct_classes += step_out["n_batch_correct_classes"]

        precision = (
            0.0
            if n_total_pr_classes == 0
            else n_total_correct_classes / n_total_pr_classes
        )
        recall = (
            0.0
            if n_total_gt_classes == 0
            else n_total_correct_classes / n_total_gt_classes
        )
        f1 = (
            0.0
            if recall * precision == 0
            else 2.0 * recall * precision / (recall + precision)
        )

        self.log_dict(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            sync_dist=True,
        )

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)
        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def _get_optimizer(self):
        opt_cfg = self.cfg.train.optimizer
        method = opt_cfg.method.lower()

        if method not in self.optimizer_types:
            raise ValueError(f"Unknown optimizer method={method}")

        kwargs = dict(opt_cfg.params)
        kwargs["params"] = self.model.parameters()
        optimizer = self.optimizer_types[method](**kwargs)

        return optimizer

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

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.cfg.workspace) / self.cfg.exp_name / self.cfg.exp_version
        model_save_path = (
            Path(self.cfg.workspace)
            / self.cfg.exp_name
            / self.cfg.exp_version
            / "huggingface_model"
        )
        tokenizer_save_path = (
            Path(self.cfg.workspace)
            / self.cfg.exp_name
            / self.cfg.exp_version
            / "huggingface_tokenizer"
        )
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(tokenizer_save_path)


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

    # prepare SROIE dataset
    train_dataset = SROIEBIODataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        mode="train",
    )
    val_dataset = SROIEBIODataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        mode="val",
    )

    # make data module & update data_module train and val dataset
    data_module = BROSDataPLModule(cfg)
    data_module.train_dataset = train_dataset
    data_module.val_dataset = val_dataset

    # Load BROS config & pretrained model
    ## update config
    bros_config = BrosConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
    bio_class_names = train_dataset.bio_class_names
    id2label = {idx: name for idx, name in enumerate(bio_class_names)}
    label2id = {name: idx for idx, name in id2label.items()}
    bros_config.id2label = id2label
    bros_config.label2id = label2id

    ## load pretrained model
    bros_model = BrosForTokenClassification.from_pretrained(
        cfg.model.pretrained_model_name_or_path, config=bros_config
    )

    # model module setting
    model_module = BROSModelPLModule(cfg, tokenizer=tokenizer)
    model_module.model = bros_model
    model_module.class_names = train_dataset.class_names
    model_module.bio_class_names = train_dataset.bio_class_names

    # define trainer logger, callbacks
    loggers = TensorBoardLogger(
        save_dir=cfg.workspace,
        name=cfg.exp_name,
        version=cfg.exp_version,
        default_hp_metric=False,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.workspace) / cfg.exp_name / cfg.exp_version / "checkpoints",
        filename="bros-sroie-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,  # if you save more than 1 model,
        # then checkpoint and huggingface model are not guaranteed to be matching
        # because we are saving with huggingface model with save_pretrained method
        # in "on_save_checkpoint" method in "BROSModelPLModule"
    )

    model_summary_callback = ModelSummary(max_depth=5)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="max")

    # define Trainer and start training
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        strategy="ddp_find_unused_parameters_true",
        num_nodes=cfg.get("num_nodes", 1),
        precision=16 if cfg.train.use_fp16 else 32,
        logger=loggers,
        callbacks=[
            lr_callback,
            checkpoint_callback,
            model_summary_callback,
            early_stop_callback,
        ],
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=3,
        gradient_clip_val=cfg.train.clip_gradient_value,
        gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
    )

    trainer.fit(model_module, data_module)


@torch.no_grad()
def inference(cfg):
    finetuned_model_path = "finetune_sroie_ee_bio/bros-base-uncased_from_dict_config3/20230721_010145/huggingface_model"
    tokenizer_path = "finetune_sroie_ee_bio/bros-base-uncased_from_dict_config3/20230721_010145/huggingface_tokenizer"

    # Load Tokenizer (going to be used in dataset to to convert texts to input_ids)
    model = BrosForTokenClassification.from_pretrained(finetuned_model_path)
    tokenizer = BrosTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path
    )

    device = "cuda"

    dataset = SROIEBIODataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        mode="val",
    )
    class_names = dataset.class_names

    # test_results = trainer.test(model=model, dataloaders = test_dataloader, ckpt_path="<your_model_checkpoint>")

    if torch.cuda.is_available():
        # model.half()
        model.to(device)
    model.eval()

    idx2bio_class = {idx: cls for idx, cls in enumerate(dataset.bio_class_names)}
    total_loss = 0

    for sample_idx, sample in tqdm(enumerate(dataset), total=len(dataset)):

        # unpack batch
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        bbox = sample["bbox"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        are_box_first_tokens = sample["are_box_first_tokens"].unsqueeze(0).to(device)
        are_box_end_tokens  = sample["are_box_end_tokens"].unsqueeze(0).to(device)
        gt_labels = sample["bio_labels"].unsqueeze(0).to(device)



        # Not necessary to use box_first_token_mask and labels when you inference unless you want get loss
        prediction = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            box_first_token_mask=are_box_first_tokens,
            labels=gt_labels,
        )

        test_loss = prediction.loss
        total_loss += test_loss
        pred_labels = torch.argmax(prediction.logits, -1)

        n_batch_gt_classes, n_batch_pred_classes, n_batch_correct_classes = 0, 0, 0
        batch_size = prediction.logits.shape[0]

        for example_idx, (pred_label, gt_label, box_first_token_mask, box_end_token_mask) in enumerate(
            zip(pred_labels, gt_labels, are_box_first_tokens, are_box_end_tokens)
        ):


            # validation loss : # calculate validation loss of "box_first_tokens" only
            valid_gt_label = gt_label[box_first_token_mask]
            valid_pred_label = pred_label[box_first_token_mask]

            gt_parse = parse_from_seq(valid_gt_label, class_names)
            pred_parse = parse_from_seq(valid_pred_label, class_names)

            """
            (Pdb++) valid_gt_label
            tensor([3, 4, 4, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')

            --> after parse

            (Pdb++) gt_parse
            [{(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)}, {(0, 1, 2, 3)}, {(45,)}, {(113,)}]
            """


            n_gt_classes = sum([len(gt_parse[class_idx]) for class_idx in range(len(class_names))])
            n_pred_classes = sum([len(pred_parse[class_idx]) for class_idx in range(len(class_names))])
            n_correct_classes = sum(
                [len(gt_parse[class_idx] & pred_parse[class_idx]) for class_idx in range(len(class_names))]
            )
            n_batch_gt_classes += n_gt_classes
            n_batch_pred_classes += n_pred_classes
            n_batch_correct_classes += n_correct_classes



            box_first_token_idx2ori_idx = box_first_token_mask.nonzero(as_tuple=True)[0]
            box2token_span_maps = torch.hstack((
                (box_first_token_mask == True).nonzero(),
                (box_end_token_mask == True).nonzero()
            )).cpu().numpy()
            start_token_idx2end_token_idx = {e[0]:e[1] for e in box2token_span_maps}

            pred_cls2text = {name: [] for name in class_names}
            gt_cls2text = deepcopy(pred_cls2text)
            correct_cls2text = deepcopy(pred_cls2text)
            incorrect_cls2text = deepcopy(pred_cls2text)
            for cls_idx, cls_name in enumerate(class_names):
                # all pred text for cls
                for box_first_token_indices in pred_parse[cls_idx]:
                    ori_indices = box_first_token_idx2ori_idx[torch.tensor(box_first_token_indices)].cpu().tolist()
                    text_span = torch.tensor(list(range(ori_indices[0], start_token_idx2end_token_idx[ori_indices[-1]])))
                    pred_text = tokenizer.decode(input_ids[example_idx][text_span])
                    pred_cls2text[cls_name].append(pred_text)

                # all gt text for cls
                for box_first_token_indices in gt_parse[cls_idx]:
                    ori_indices = box_first_token_idx2ori_idx[torch.tensor(box_first_token_indices)].cpu().tolist()
                    text_span = torch.tensor(list(range(ori_indices[0], start_token_idx2end_token_idx[ori_indices[-1]])))
                    gt_text = tokenizer.decode(input_ids[example_idx][text_span])
                    gt_cls2text[cls_name].append(gt_text)

                # all correct text for cls
                for box_first_token_indices in pred_parse[cls_idx] & gt_parse[cls_idx]:
                    ori_indices = box_first_token_idx2ori_idx[torch.tensor(box_first_token_indices)].cpu().tolist()
                    text_span = torch.tensor(list(range(ori_indices[0], start_token_idx2end_token_idx[ori_indices[-1]])))
                    correct_text = tokenizer.decode(input_ids[example_idx][text_span])
                    correct_cls2text[cls_name].append(correct_text)

                # all incorrect text for cls (text in gt but not in pred + text not in gt but in pred)
                for box_first_token_indices in pred_parse[cls_idx] ^ gt_parse[cls_idx]:
                    ori_indices = box_first_token_idx2ori_idx[torch.tensor(box_first_token_indices)].cpu().tolist()
                    text_span = torch.tensor(list(range(ori_indices[0], start_token_idx2end_token_idx[ori_indices[-1]])))
                    incorrect_text = tokenizer.decode(input_ids[example_idx][text_span])
                    incorrect_cls2text[cls_name].append(incorrect_text)



        print(f"{pred_cls2text = }")
        print(f"{gt_cls2text = }")
        print(f"{correct_cls2text = }")
        print(f"{incorrect_cls2text = }")
    total_loss = total_loss / len(dataset)
    print(f"{total_loss = }")

if __name__ == "__main__":
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
    cfg = OmegaConf.create(finetune_sroie_ee_bio_config)
    # train(cfg)
    inference(cfg)
