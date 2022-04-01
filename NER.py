# -*- coding: UTF-8 -*-
"""
@Project ：entity recognition 
@File ：NER.py
@Author ：AnthonyZ
@Date ：2022/3/31 23:49
"""

import torch
import torch.nn as nn
import os


class CharCNN(nn.Module):
    def __init__(self,
                 w_emb_dim=200,
                 word_maxlen=30,
                 sent_maxlen=35,
                 num_train=4000,
                 num_dev=1000,
                 num_test=3257,
                 num_sent=8257):
        super(CharCNN, self).__init__()
        #
        # c_emb_dim = 30
        # w_emb_dim_char_level = 50
        # final_w_emb_dim = 200
        #
        # nb_epoch = 25
        # batch_size = 10
        #
        # feat_dim = 512
        # w = 7
        # num_region = 49
        # self.emb = nn.Embedding(num_embeddings=, embedding_dim=w_emb_dim, padding_idx=0)


# class bi_LSTM(nn.Module):
#     def __init__(self):
#         super(bi_LSTM, self).__init__()
#         self.emb = nn.Embedding()





if __name__ == "__main__":
    a, b, c, d, e = load_vocab()