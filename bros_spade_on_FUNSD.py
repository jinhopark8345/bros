# PYTHONPATH=/home/jinho/Projects/bros python3 funsd_bioes_getitem_demo.py

from pprint import pprint
from datasets import load_dataset
from bros import BrosConfig, BrosTokenizer
import itertools
import numpy as np
from torch.utils.data.dataset import Dataset
from dataclasses import dataclass
import torch
# from transformers import AutoTokenizer

torch.set_printoptions(threshold=2000000)

class FUNSDSpadeDataset(Dataset):
    """ FUNSD BIOES tagging Dataset

    FUNSD : Form Understanding in Noisy Scanned Documents
    BIOES tagging : begin, in, out, end, single tagging

    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_length=512,
        split='train',
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

        self.class_names = ['other', 'header', 'question', 'answer']
        self.out_class_name = 'other'
        self.class_idx_dic = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.pad_token = self.tokenizer.pad_token
        self.ignore_label_id = -100

    def __len__(self):
        return len(self.examples)

    def tokenize_word_for_spade(self, word):
        bboxes = [e['box'] for e in word]
        texts = [e['text'] for e in word]

        word_input_ids = []
        word_bboxes = []
        for idx, (bbox, text) in enumerate(zip(bboxes, texts)):
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            word_input_ids.append(input_ids)
            word_bboxes.append([bbox for _ in range(len(input_ids))])

        if len(word_input_ids) <= 0:
            breakpoint()
        assert len(word_input_ids) > 0
        assert len(word_input_ids) == len(word_bboxes)

        return word_input_ids, word_bboxes

    def __getitem__(self, idx):
        sample = self.examples[idx]

        word_labels = sample['labels']
        words = sample['words']
        assert len(word_labels) == len(words)

        width, height = sample['img'].size
        cls_bbs = [0] * 4 # bbox for first token
        sep_bbs = [width, height] * 2 # bbox for last token

        padded_input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        padded_bboxes = np.zeros((self.max_seq_length, 4), dtype=np.float32)
        padded_labels = np.ones(self.max_seq_length, dtype=int) * -100
        attention_mask = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        are_box_end_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)


        input_ids_list: List[List[int]] = []
        labels_list: List[List[str]] = []
        bboxes_list: List[List[List[int]]] = []
        start_token_indices = []
        end_token_indices = []


        # # filter out word with empty text
        word_and_label_list = []
        for word, label in zip(words, word_labels):
            cur_word_and_label = []
            for e in word:
                if e['text'].strip() != '':
                    cur_word_and_label.append(e)
            if cur_word_and_label:
                word_and_label_list.append((cur_word_and_label, label))

        # save classes_dic
        text_box_idx = 0
        label2text_box_indices_list = {cls_name: [] for cls_name in self.class_names}
        for (word, label) in word_and_label_list:
            if label == self.out_class_name:
                text_box_idx += len(word)
                continue

            text_box_indices = []
            for text_and_box in word:
                text_box_indices.append(text_box_idx)
                text_box_idx += 1

            label2text_box_indices_list[label].append(text_box_indices)


        input_ids = []
        bboxes = []
        box_to_token_indices = []
        cum_token_idx = 0

        text_box_list = [text_and_box for (word, label) in word_and_label_list for text_and_box in word]
        for text_box in text_box_list:
            text, box = text_box['text'], text_box['box']
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

            box_to_token_indices.append(this_box_token_indices)

        # For [CLS] and [SEP]
        input_ids = [self.cls_token_id] + input_ids[: self.max_seq_length - 2] + [self.sep_token_id]
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

        padded_input_ids = torch.from_numpy(padded_input_ids)
        padded_bboxes = torch.from_numpy(padded_bboxes)
        attention_mask = torch.from_numpy(attention_mask)

        ## need to work on making itc labels and stc labels!!!!!!!


        breakpoint()




        # return_dict = {
        #     "input_ids": padded_input_ids,
        #     "bbox": padded_bboxes,
        #     "attention_mask": attention_mask,
        #     "are_box_first_tokens": are_box_first_tokens,
        #     "are_box_end_tokens": are_box_end_tokens,
        #     "labels": padded_labels,
        # }

        # breakpoint()


        return ""



@dataclass
class CFG:
    dataset: str
    max_seq_length: dict
    tokenizer_path: str


if __name__ == '__main__':


    cfg = CFG("jinho8345/funsd", 512, "naver-clova-ocr/bros-base-uncased")

    # Load Tokenizer (going to be used in dataset to to convert texts to input_ids)
    tokenizer = BrosTokenizer.from_pretrained(cfg.tokenizer_path)

    # prepare SROIE dataset
    train_dataset = FUNSDSpadeDataset(
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
