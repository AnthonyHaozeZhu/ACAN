# -*- coding: UTF-8 -*-
"""
@Project ：entity recognition 
@File ：NER.py
@Author ：AnthonyZ
@Date ：2022/3/31 23:49
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(padding_idx=0, num_embeddings=1000, embedding_dim=30)  # 相当于创建符号的字母表
        nn.init.uniform_(self.embedding.weight, -0.25, 0.25)

        self.conv_1 = nn.Conv1d(in_channels=30, out_channels=32, kernel_size=2, padding=1)
        self.pooling_1 = nn.MaxPool1d(kernel_size=2)

        self.conv_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pooling_2 = nn.MaxPool1d(kernel_size=2)

        self.conv_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, padding=2)
        self.pooling_3 = nn.MaxPool1d(kernel_size=2)

        self.batch_normal = nn.BatchNorm1d(32)

        self.Linear_1 = nn.Linear(128, 100)
        self.Linear_2 = nn.Linear(100, 50)

    def forward(self, x):
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)
        x = self.embedding(x)  # (b, s, w, d)
        x = x.view(batch_size * max_seq_len, max_word_len, -1)  # (b*s, w, d)
        x = x.transpose(2, 1)  # (b*s, d, w): Conv1d takes in (batch, dim, seq_len), but raw embedded is (batch, seq_len, dim)

        x = self.conv_1(x)
        x = F.relu(x)
        x = self.pooling_1(x)
        x = F.dropout(x, 0.25)

        x = self.conv_2(x)
        x = F.relu(x)
        x = self.pooling_2(x)
        x = F.dropout(x, 0.25)

        x = self.conv_3(x)
        x = F.relu(x)
        x = self.pooling_3(x)
        x = F.dropout(x, 0.25)

        x = self.batch_normal(x)

        x = x.view(x.size(0), -1)

        x = self.Linear_1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.Linear_2(x)

        return x





