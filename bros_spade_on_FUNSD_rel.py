import itertools
from dataclasses import dataclass
from pprint import pprint

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from bros import BrosConfig, BrosTokenizer
from datasets import load_dataset

# from transformers import AutoTokenizer

torch.set_printoptions(threshold=2000000)


"""

Entity Linking (EL) task is a task linking between entities.
For example,  "DATE" text is connected to "12/12/2019" text.

    "DATE" -> tokens : ["DA", "TE"], input_ids : [3312, 5123]
    "12/12/2019" -> tokens : ["12", "/", "12", "/", "20", "19"], input_ids : [12, 13, 12, 13, 1123, 777]

    here you are training a model to connect "DA" token with "12" token

However, in Entity Extraction (EE) task is a task to extract entities
from sequence of text.  Often times, Entities consist of multiple
words and each word consists of multiple tokens.  So in EE task, you
are finding entities by finding corressponding tokens (or connecting
corresponding tokens that belong to each entity)

"""

class FUNSDSpadeRelDataset(Dataset):
    """FUNSD BIOES tagging Dataset

    FUNSD : Form Understanding in Noisy Scanned Documents

    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_length=512,
        split="train",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        self.examples = load_dataset(self.dataset)[split]

        self.class_names = ["other", "header", "question", "answer"]
        self.out_class_name = "other"
        self.class_idx_dic = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }
        self.pad_token = self.tokenizer.pad_token
        self.ignore_label_id = -100

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]

        word_labels = sample["labels"]
        words = sample["words"]
        linkings = sample["linkings"]
        assert len(word_labels) == len(words)

        width, height = sample["img"].size
        cls_bbs = [0] * 4  # bbox for first token
        sep_bbs = [width, height] * 2  # bbox for last token

        # make placeholders
        padded_input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        padded_bboxes = np.zeros((self.max_seq_length, 4), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        are_box_end_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        itc_labels = np.zeros(self.max_seq_length, dtype=int)
        stc_labels = np.ones(self.max_seq_length, dtype=np.int64) * self.max_seq_length
        el_labels = np.ones(self.max_seq_length, dtype=int) * self.max_seq_length

        # convert linkings from "word_idx to word_idx" to "text_box to text_box"
        from_text_box2to_text_box = {}
        for linking in linkings:
            if not linking:
                continue
            from_word_idx, to_word_idx = linking[0]
            from_text_box, to_text_box = words[from_word_idx][0], words[to_word_idx][0]
            from_text_box = tuple([from_text_box["text"], tuple(from_text_box["box"])])
            to_text_box = tuple([to_text_box["text"], tuple(to_text_box["box"])])
            from_text_box2to_text_box[from_text_box] = to_text_box

        """
        in the beginning, words are like below

            [
                [{'box': [147, 148, 213, 168], 'text': 'Attorney'},
                    {'box': [216, 151, 275, 168], 'text': 'General'},
                    {'box': [148, 172, 187, 190], 'text': 'Betty'},
                    {'box': [191, 169, 206, 187], 'text': 'D.'},
                    {'box': [211, 170, 305, 191], 'text': 'Montgomery'}],
                [{'box': [275, 249, 377, 267], 'text': 'CONFIDENTIAL'},
                    {'box': [380, 250, 457, 267], 'text': 'FACSIMILE'},
                    {'box': [264, 267, 369, 281], 'text': 'TRANSMISSION'},
                    {'box': [369, 267, 422, 281], 'text': 'COVER'},
                    {'box': [420, 267, 467, 281], 'text': 'SHEET'}],
                [{'box': [352, 297, 383, 314], 'text': '(614)'},
                    {'box': [384, 296, 405, 313], 'text': '466-'},
                    {'box': [406, 297, 438, 312], 'text': '5087'}]
                ...
            ]

        and "words" and "word_labels" are synchronized based on their indices

        1. filter out "text_box" with emtpy text
        2. convert words into input_ids and bboxes
            2-1. convert word into "list of text_box"
                2-1-1. convert "text_box" into "list of tokens"

        in result we will have,

            - input_ids & bboxes are synchronized based on their indices

            - text_box_idx2token_indices : List[List[int]]:
                    -> text_box_idx to token_indices (of corressponding text)

            - label2text_box_indices_list : Dict[str, List[List[int]]]
                    -> list of text_box_indices belong to each label (class_name) mapping

            - text_box2text_box_idx : Dict[tuple, int]
                    -> tuple value of text_box to text_box_idx mapping,
                       going to use with "from_text_box2to_text_box" (came from converting linking gt)
                       to get linkings between text_box_indices

        """

        # 1. filter out "text_box" with emtpy text
        word_and_label_list = []
        for word, label in zip(words, word_labels):
            cur_word_and_label = []
            for e in word:
                if e["text"].strip() != "":
                    cur_word_and_label.append(e)
            if cur_word_and_label:
                word_and_label_list.append((cur_word_and_label, label))


        # 2. convert words into input_ids and bboxes
        text_box_idx = 0
        cum_token_idx = 0
        input_ids = []
        bboxes = []
        text_box_idx2token_indices = []
        label2text_box_indices_list = {cls_name: [] for cls_name in self.class_names}
        text_box2text_box_idx = {}
        for word_idx, (word, label) in enumerate(word_and_label_list):
            text_box_indices = []
            for text_and_box in word:
                text_box_indices.append(text_box_idx)

                text, box = text_and_box["text"], text_and_box["box"]
                text_box2text_box_idx[tuple([text, tuple(box)])] = text_box_idx
                this_box_token_indices = []

                if text.strip() == "":
                    continue

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                input_ids += tokens
                bb = [box for _ in range(len(tokens))]
                bboxes += bb

                for _ in tokens:
                    cum_token_idx += 1
                    this_box_token_indices.append(cum_token_idx)

                text_box_idx2token_indices.append(this_box_token_indices)
                text_box_idx += 1

            label2text_box_indices_list[label].append(text_box_indices)
        tokens_length_list: List[int] = [len(l) for l in label2text_box_indices_list]

        # convert linkings from "text_box to text_box" to "text_box idx to text_box idx"
        from_text_box_idx2to_text_box_idx = {
            text_box2text_box_idx[from_text_box]: text_box2text_box_idx[to_text_box]
            for from_text_box, to_text_box in from_text_box2to_text_box.items()
        }

        # consider [CLS] token that will be added to input_ids, shift "end token indices" 1 to the right
        et_indices = np.array(list(itertools.accumulate(tokens_length_list))) + 1

        # since we subtract original length from shifted indices, "start token indices" are aligned as well
        st_indices = et_indices - np.array(tokens_length_list)

        # last index will be used for [SEP] token
        # to make sure st_indices and end_indices are paired, in case st_indices are cut by max_sequence length,
        st_indices = st_indices[st_indices < self.max_seq_length - 1]
        et_indices = et_indices[et_indices < self.max_seq_length - 1]

        # to make sure st_indices and end_indices are paired, in case st_indices are cut by max_sequence length,
        min_len = min(len(st_indices), len(et_indices))
        st_indices = st_indices[:min_len]
        et_indices = et_indices[:min_len]
        assert len(st_indices) == len(et_indices)

        are_box_first_tokens[st_indices] = True
        are_box_end_tokens[et_indices] = True

        # from_text_box_idx2to_text_box_idx = {k-1: v-1 for k, v in from_text_box_idx2to_text_box_idx.items()}
        for from_idx, to_idx in from_text_box_idx2to_text_box_idx.items():

            if from_idx >= len(text_box_idx2token_indices) or to_idx >= len(text_box_idx2token_indices):
                continue

            if (
                text_box_idx2token_indices[from_idx][0] >= self.max_seq_length
                or text_box_idx2token_indices[to_idx][0] >= self.max_seq_length
            ):
                continue

            word_from = text_box_idx2token_indices[from_idx][0]
            word_to = text_box_idx2token_indices[to_idx][0]
            el_labels[word_to] = word_from


        # For [CLS] and [SEP]
        input_ids = (
            [self.cls_token_id]
            + input_ids[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(bboxes) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            bboxes = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            bboxes = [cls_bbs] + bboxes[: self.max_seq_length - 2] + [sep_bbs]
        bboxes = np.array(bboxes)

        # update ppadded input_ids, labels, bboxes
        len_ori_input_ids = len(input_ids)
        padded_input_ids[:len_ori_input_ids] = input_ids
        # padded_labels[:len_ori_input_ids] = np.array(labels)
        attention_mask[:len_ori_input_ids] = 1
        padded_bboxes[:len_ori_input_ids, :] = bboxes

        # expand bbox from [x1, y1, x2, y2] (2points) -> [x1, y1, x2, y1, x2, y2, x1, y2] (4points)
        padded_bboxes = padded_bboxes[:, [0, 1, 2, 1, 2, 3, 0, 3]]

        # Normalize bbox -> 0 ~ 1
        padded_bboxes[:, [0, 2, 4, 6]] = padded_bboxes[:, [0, 2, 4, 6]] / width
        padded_bboxes[:, [1, 3, 5, 7]] = padded_bboxes[:, [1, 3, 5, 7]] / height

        # convert to tensor
        padded_input_ids = torch.from_numpy(padded_input_ids)
        padded_bboxes = torch.from_numpy(padded_bboxes)
        attention_mask = torch.from_numpy(attention_mask)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
        are_box_end_tokens = torch.from_numpy(are_box_end_tokens)
        itc_labels = torch.from_numpy(itc_labels)
        stc_labels = torch.from_numpy(stc_labels)
        el_labels = torch.from_numpy(el_labels)

        return_dict = {
            "input_ids": padded_input_ids,
            "bbox": padded_bboxes,
            "attention_mask": attention_mask,
            "are_box_first_tokens": are_box_first_tokens,
            "are_box_end_tokens": are_box_end_tokens,
            "el_labels": el_labels,
            "itc_labels": itc_labels,
            "stc_labels": stc_labels,
        }

        return return_dict


@dataclass
class CFG:
    dataset: str
    max_seq_length: dict
    tokenizer_path: str


if __name__ == "__main__":
    cfg = CFG("jinho8345/funsd", 512, "naver-clova-ocr/bros-base-uncased")

    # Load Tokenizer (going to be used in dataset to to convert texts to input_ids)
    tokenizer = BrosTokenizer.from_pretrained(cfg.tokenizer_path)

    # prepare SROIE dataset
    train_dataset = FUNSDSpadeRelDataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        split="test",
    )

    sample1 = train_dataset[0]
    breakpoint()
    # for s in train_dataset:
    #     tmp = s

    # for sample in d:
    #     tmp.handle_sample(sample)
    #     break
