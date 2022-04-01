# -*- coding: UTF-8 -*-
"""
@Project ：entity recognition 
@File ：load_data.py
@Author ：AnthonyZ
@Date ：2022/3/30 22:32
"""
import os
import copy
import json
import logging

import numpy as np

import torch
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, word_ids, char_ids, img_feature, mask, label_ids):
        self.word_ids = word_ids
        self.char_ids = char_ids
        self.img_feature = img_feature
        self.mask = mask
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def load_vocab():
    word_vocab_path = "./vocab/word_vocab"
    char_vocab_path = "./vocab/char_vocab"

    word_vocab = dict()
    char_vocab = dict()
    word_ids_to_tokens = []
    char_ids_to_tokens = []

    # Load word vocab
    with open(word_vocab_path, "r", encoding="utf-8") as f:
        # Set the exact vocab size
        # If the original vocab size is smaller than args.vocab_size, then set args.vocab_size to original one
        word_lines = f.readlines()
        word_vocab_size = len(word_lines)

        for idx, line in enumerate(word_lines[:word_vocab_size]):
            line = line.strip()
            word_vocab[line] = idx
            word_ids_to_tokens.append(line)

    # Load char vocab
    with open(char_vocab_path, "r", encoding="utf-8") as f:
        char_lines = f.readlines()
        char_vocab_size = len(char_lines)
        for idx, line in enumerate(char_lines[:char_vocab_size]):
            line = line.strip()
            char_vocab[line] = idx
            char_ids_to_tokens.append(line)

    return word_vocab, char_vocab, word_ids_to_tokens, word_vocab_size, char_vocab_size


def load_word_matrix(word_vocab, word_vocab_size, word_emb_dim=200):
    embedding_index = dict()
    with open(os.path.join("./wordvec/word_vector_200d.vec"), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    word_matrix = np.zeros((word_vocab_size, word_emb_dim), dtype='float32')
    cnt = 0

    for word, i in word_vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            word_matrix[i] = embedding_vector
        else:
            word_matrix[i] = np.random.uniform(-0.25, 0.25, word_emb_dim)
            cnt += 1
    logger.info('{} words not in pretrained matrix'.format(cnt))

    word_matrix = torch.from_numpy(word_matrix)
    return word_matrix


def convert_examples_to_features(examples,
                                 img_features,
                                 max_seq_len,
                                 max_word_len,
                                 word_vocab,
                                 char_vocab,
                                 label_vocab,
                                 pad_token="[pad]",
                                 unk_token="[unk]"):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 1. Load img feature
        try:
            img_feature = img_features[example.img_id]
        except:
            logger.warning("Cannot load image feature! (IMGID: {})".format(example.img_id))
            continue

        # 2. Convert tokens to idx & Padding
        word_pad_idx, char_pad_idx, label_pad_idx = word_vocab[pad_token], char_vocab[pad_token], label_vocab[pad_token]
        word_unk_idx, char_unk_idx, label_unk_idx = word_vocab[unk_token], char_vocab[unk_token], label_vocab[unk_token]

        word_ids = []
        char_ids = []
        label_ids = []

        for word in example.words:
            word_ids.append(word_vocab.get(word, word_unk_idx))
            ch_in_word = []
            for char in word:
                ch_in_word.append(char_vocab.get(char, char_unk_idx))
            # Padding for char
            char_padding_length = max_word_len - len(ch_in_word)
            ch_in_word = ch_in_word + ([char_pad_idx] * char_padding_length)
            ch_in_word = ch_in_word[:max_word_len]
            char_ids.append(ch_in_word)

        for label in example.labels:
            label_ids.append(label_vocab.get(label, label_unk_idx))

        mask = [1] * len(word_ids)

        # Padding for word and label
        word_padding_length = max_seq_len - len(word_ids)
        word_ids = word_ids + ([word_pad_idx] * word_padding_length)
        label_ids = label_ids + ([label_pad_idx] * word_padding_length)
        mask = mask + ([0] * word_padding_length)

        word_ids = word_ids[:max_seq_len]
        label_ids = label_ids[:max_seq_len]
        char_ids = char_ids[:max_seq_len]
        mask = mask[:max_seq_len]

        # Additional padding for char if word_padding_length > 0
        if word_padding_length > 0:
            for i in range(word_padding_length):
                char_ids.append([char_pad_idx] * max_word_len)

        # 3. Verify
        assert len(word_ids) == max_seq_len, "Error with word_ids length {} vs {}".format(len(word_ids), max_seq_len)
        assert len(char_ids) == max_seq_len, "Error with char_ids length {} vs {}".format(len(char_ids), max_seq_len)
        assert len(label_ids) == max_seq_len, "Error with label_ids length {} vs {}".format(len(label_ids), max_seq_len)
        assert len(mask) == max_seq_len, "Error with mask length {} vs {}".format(len(mask), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("img_id: %s" % example.img_id)
            logger.info("words: %s" % " ".join([str(x) for x in example.words]))
            logger.info("word_ids: %s" % " ".join([str(x) for x in word_ids]))
            logger.info("char_ids[0]: %s" % " ".join([str(x) for x in char_ids[0]]))
            logger.info("mask: %s" % " ".join([str(x) for x in mask]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(word_ids=word_ids,
                          char_ids=char_ids,
                          img_feature=img_feature,
                          mask=mask,
                          label_ids=label_ids
                          ))

    return features




if __name__ == "__main__":
    a, b, c, d, e = load_vocab()
    mm = load_word_matrix(a, d)
