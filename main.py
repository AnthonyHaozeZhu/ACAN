# -*- coding: UTF-8 -*-
"""
@Project ：entity recognition 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/4/1 19:43
"""
import argparse

from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import NER

from utils import init_logger, build_vocab, download_vgg_features, set_seed, load_vocab
from load_data import load_data, load_word_matrix


def main(args):
    init_logger()
    set_seed(args)
    download_vgg_features(args)
    build_vocab(args)

    train_dataset = load_data(args, mode="train")
    dev_dataset = load_data(args, mode="dev")
    test_dataset = load_data(args, mode="test")

    return train_dataset, dev_dataset, test_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path for saving model")
    parser.add_argument("--wordvec_dir", default="./wordvec", type=str, help="Path for pretrained word vector")
    parser.add_argument("--vocab_dir", default="./vocab", type=str)

    parser.add_argument("--train_file", default="train", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test", type=str, help="Test file")
    parser.add_argument("--w2v_file", default="word_vector_200d.vec", type=str, help="Pretrained word vector file")
    parser.add_argument("--img_feature_file", default="img_vgg_features.pt", type=str, help="Filename for preprocessed image features")

    parser.add_argument("--max_seq_len", default=35, type=int, help="Max sentence length")
    parser.add_argument("--max_word_len", default=30, type=int, help="Max word length")

    parser.add_argument("--word_vocab_size", default=23204, type=int, help="Maximum size of word vocabulary")
    parser.add_argument("--char_vocab_size", default=102, type=int, help="Maximum size of character vocabulary")

    parser.add_argument("--word_emb_dim", default=200, type=int, help="Word embedding size")
    parser.add_argument("--char_emb_dim", default=30, type=int, help="Character embedding size")
    parser.add_argument("--final_char_dim", default=50, type=int, help="Dimension of character cnn output")
    parser.add_argument("--hidden_dim", default=200, type=int, help="Dimension of BiLSTM output, att layer (denoted as k) etc.")

    parser.add_argument("--kernel_lst", default="2,3,4", type=str, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")

    parser.add_argument('--seed', type=int, default=7, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation")
    # parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--slot_pad_label", default="[pad]", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")
    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--logging_steps', type=int, default=250, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=250, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--no_w2v", action="store_true", help="Not loading pretrained word vector")

    args = parser.parse_args()

    # For VGG16 img features (DO NOT change this part)
    args.num_img_region = 49
    args.img_feat_dim = 512

    train_dataset = load_data(
        args,
        mode="train")
    dev_dataset = load_data(
        args,
        mode="dev")
    test_dataset = load_data(
        args,
        mode="test")
    model = NER.CharCNN()
    train_sampler = RandomSampler(
        train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size)
    # a = model.forward(train_dataset)
    word_vocab, char_vocab, word_ids_to_tokens, char_ids_to_tokens = load_vocab(args)
    pretrained_word_matrix = load_word_matrix(
        args,
        word_vocab)
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to('cpu')for t in batch)  # GPU or CPU
        a = batch[0]
        b = batch[1]
        c = batch[2]
        d = batch[3]
        e = batch[4]
    # tt = NER.bi_LSTM(pretrained_word_matrix)
    # a = tt()

    net = NER.ACN(pretrained_word_matrix)
    m, n = net(a, b, c, d, e)



