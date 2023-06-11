"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0

Do 2nd preprocess on top of the result of the 'preprocess.sh' file.
Reference: https://github.com/microsoft/unilm/blob/master/layoutlm/deprecated/examples/seq_labeling/run_seq_labeling.py
"""


import json
import os
from collections import Counter
from datasets import load_dataset

from tqdm import tqdm
from transformers import BertTokenizer

MAX_SEQ_LENGTH = 4000
MODEL_TYPE = "bert"
VOCA = "bert-base-uncased"

INPUT_PATH = "./data"
OUTPUT_PATH = "../../datasets/funsd"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "preprocessed"), exist_ok=True)

max_len_poss = -1

def main():
    for dataset_split in ["train", "val"]:
        print(f"dataset_split: {dataset_split}")
        do_2nd_preprocess(dataset_split)

    os.system(f"cp -r {os.path.join(INPUT_PATH, 'training_data')} {OUTPUT_PATH}")
    os.system(f"cp -r {os.path.join(INPUT_PATH, 'validation_data')} {OUTPUT_PATH}")
    os.system(f"cp {os.path.join(INPUT_PATH, 'labels.txt')} {OUTPUT_PATH}")


def do_2nd_preprocess(dataset_split):
    label_fpath = os.path.join(INPUT_PATH, "labels.txt")
    labels = get_labels(label_fpath)

    tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    ignore_index = -100

    examples = load_dataset("jinho8345/bros-funsd-bies")[dataset_split]
    # examples = read_examples_from_file(INPUT_PATH, mode)

    breakpoint()
    features = convert_examples_to_features(
        examples,
        labels,
        MAX_SEQ_LENGTH,
        tokenizer,
        cls_token_at_end=bool(MODEL_TYPE in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if MODEL_TYPE in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(MODEL_TYPE in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(MODEL_TYPE in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if MODEL_TYPE in ["xlnet"] else 0,
        pad_token_label_id=ignore_index,
    )

    # Save image ocr files
    image_cnter = Counter()
    preprocessed_fnames = []
    for example, feature in tqdm(zip(examples, features), total=len(examples)):
        # Example: guid, words, labels, boxes, actual_bboxes, file_name, page_size
        # Feature: input_ids, input_mask, segment_ids, label_ids,
        #          boxes, actual_bboxes, file_name, page_size

        file_name = example['file_name']
        page_size = example['page_size']

        this_file_name = "{}_{}.json".format(
            file_name[: file_name.rfind(".")],
            image_cnter[file_name],
        )
        image_cnter[file_name] += 1

        data_obj = {}

        # meta
        data_obj["meta"] = {}
        # data_obj["meta"]["image_size"]
        #     = example.page_size[::-1] + [3]  # [height, width, rgb?]
        height, width = page_size[::-1]
        data_obj["meta"]["imageSize"] = {"width": width, "height": height}
        data_obj["meta"]["voca"] = VOCA

        if dataset_split == "train":
            data_obj["meta"]["image_path"] = os.path.join(
                "training_data", "images", file_name
            )
        elif dataset_split == "val":
            data_obj["meta"]["image_path"] = os.path.join(
                "validation_data", "images", file_name
            )

        # words
        #   text, tokens, boundingBox

        words = example['words']
        actual_bboxes = example['actual_bboxes']


        data_obj["words"] = []
        this_input_ids = []
        for word, bb in zip(words, actual_bboxes):
            word_tokens = []
            for splitted_word in word.split():
                word_tokens.append(
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(splitted_word))
                )

            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))

            word_obj = {
                "text": word,
                "tokens": tokens,
                "boundingBox": [
                    [bb[0], bb[1]],
                    [bb[2], bb[1]],
                    [bb[2], bb[3]],
                    [bb[0], bb[3]],
                ],
            }

            data_obj["words"].append(word_obj)
            this_input_ids.extend(tokens)

        if VOCA == "bert-base-uncased":
            feature_input_ids = feature.input_ids
            assert feature_input_ids[0] == cls_token_id
            feature_input_ids = feature_input_ids[
                1 : feature_input_ids.index(sep_token_id)
            ]
            assert feature_input_ids == this_input_ids
        else:
            raise NotImplementedError

        # masks, labels
        data_obj["parse"] = {}
        if VOCA == "bert-base-uncased":
            data_obj["parse"]["seq_len"] = sum(feature.input_mask)
            data_obj["parse"]["input_ids"] = feature.input_ids
            data_obj["parse"]["input_mask"] = feature.input_mask
            data_obj["parse"]["label_ids"] = feature.label_ids
        else:
            raise NotImplementedError

        # Save file name to list
        preprocessed_fnames.append(os.path.join("preprocessed", this_file_name))

        # Save to file
        data_obj_file = os.path.join(OUTPUT_PATH, "preprocessed", this_file_name)
        with open(data_obj_file, "w", encoding="utf-8") as fp:
            json.dump(data_obj, fp, ensure_ascii=False)

    # Save file name list file
    preprocessed_filelist_file = os.path.join(
        OUTPUT_PATH, f"preprocessed_files_{dataset_split}.txt"
    )
    with open(preprocessed_filelist_file, "w", encoding="utf-8") as fp:
        fp.write("\n".join(preprocessed_fnames))


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


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
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
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

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        file_name = example[ "file_name" ]
        page_size = example["page_size"]
        width, height = page_size
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
                [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
            )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2

        global max_len_poss
        ori = max_len_poss
        max_len_poss = max(max_len_poss, len(tokens))
        if ori < max_len_poss:
            print(f'{max_len_poss = }')

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
        actual_bboxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            actual_bboxes = [[0, 0, width, height]] + actual_bboxes
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
        else:
            input_ids += [pad_token] * padding_length
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


if __name__ == "__main__":
    main()

