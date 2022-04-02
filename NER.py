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
from TorchCRF import CRF

from load_data import TweetProcessor


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
        x = x.view(batch_size, max_seq_len, -1)

        return x


class bi_LSTM(nn.Module):
    def __init__(self, pretrained_word_matrix):
        super(bi_LSTM, self).__init__()
        self.CNN = CharCNN()
        if pretrained_word_matrix is not None:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_matrix)
        else:
            self.word_embedding = nn.Embedding(padding_idx=0, num_embeddings=23204, embedding_dim=200)
            nn.init.uniform_(self.word_embedding.weight, -0.25, 0.25)
        self.lstm = nn.LSTM(input_size=200 + 50,
                            hidden_size=100,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, word_ids, char_ids):
        word_embedding = self.word_embedding(word_ids)
        char_embedding = self.CNN(char_ids)
        union_embedding = torch.cat([word_embedding, char_embedding], dim=-1)
        x, _ = self.lstm(union_embedding, None)

        return x


class Co_Attention(nn.Module):
    def __init__(self):
        super(Co_Attention, self).__init__()
        self.text_Linear_1 = nn.Linear(200, 200, bias=True)
        self.img_Linear_1 = nn.Linear(200, 200, bias=False)
        self.Linear_1 = nn.Linear(400, 1)

        self.text_Linear_2 = nn.Linear(200, 200, bias=False)
        self.img_Linear_2 = nn.Linear(200, 200, bias=True)
        self.Linear_2 = nn.Linear(400, 1)

    def forward(self, text_embedding, img_embedding):
        text_embedding_repeat = text_embedding.unsqueeze(2).repeat(1, 1, 49, 1)
        img_embedding_repeat = img_embedding.unsqueeze(1).repeat(1, 35, 1, 1)
        text_embedding_repeat = self.text_Linear_1(text_embedding_repeat)
        img_embedding_repeat = self.img_Linear_1(img_embedding_repeat)
        union_feature = torch.cat((text_embedding_repeat, img_embedding_repeat), dim=-1)
        union_feature = torch.tanh(union_feature)
        union_feature = self.Linear_1(union_feature).squeeze(-1)
        visial_att = F.softmax(union_feature, dim=-1)
        att_img_feature = torch.matmul(visial_att, img_embedding)

        text_embedding_repeat = text_embedding.unsqueeze(1).repeat(1, 35, 1, 1)
        img_embedding_repeat = att_img_feature.unsqueeze(2).repeat(1, 1, 35, 1)
        text_embedding_repeat = self.text_Linear_2(text_embedding_repeat)
        img_embedding_repeat = self.img_Linear_2(img_embedding_repeat)
        union_feature = torch.cat((img_embedding_repeat, text_embedding_repeat), dim=-1)
        union_feature = torch.tanh(union_feature)
        text_att = self.Linear_2(union_feature).squeeze(-1)
        text_att = F.softmax(text_att, dim=-1)
        att_text_feature = torch.matmul(text_att, text_embedding)

        return att_text_feature, att_img_feature


class GMF(nn.Module):
    def __init__(self):
        super(GMF, self).__init__()
        self.text_Linear = nn.Linear(200, 200, bias=True)
        self.img_Linear = nn.Linear(200, 200, bias=True)
        self.Linear = nn.Linear(400, 1, bias=False)

    def forward(self, att_text_feature, att_img_feature):
        att_text_feature = self.text_Linear(att_text_feature)
        att_text_feature = torch.tanh(att_text_feature)
        att_img_feature = self.img_Linear(att_img_feature)
        att_img_feature = torch.tanh(att_img_feature)
        union_feature = torch.cat([att_img_feature, att_text_feature], dim=-1)
        g_att = self.Linear(union_feature)
        g_att = torch.sigmoid(g_att)
        g_att = g_att.repeat(1, 1, 200)
        multi_model = torch.mul(g_att, att_img_feature) + torch.mul((1 - g_att), att_text_feature)
        return multi_model


class FiltrationGate(nn.Module):
    def __init__(self):
        super(FiltrationGate, self).__init__()
        self.mult_Linear = nn.Linear(200, 200, bias=True)
        self.text_Linear = nn.Linear(200, 200, bias=False)
        self.resv_Linear = nn.Linear(200, 200, bias=True)
        self.reshape = nn.Linear(400, 1)
        self.output_Linear = nn.Linear(400, len(TweetProcessor.get_labels()))

    def forward(self, mult_model, text_embedding):
        m_model = self.mult_Linear(mult_model)
        t_embedding = self.text_Linear(text_embedding)
        concat_feature = torch.cat([t_embedding, m_model], dim=-1)
        filtration_gate = torch.sigmoid(concat_feature)
        filtration_gate = self.reshape(filtration_gate)
        filtration_gate = torch.sigmoid(filtration_gate)
        filtration_gate = filtration_gate.repeat(1, 1, 200)
        mult_model = self.resv_Linear(mult_model)
        mult_model = torch.tanh(mult_model)
        reserved_multimodal_feat = torch.mul(filtration_gate, mult_model)
        output = torch.cat((text_embedding, reserved_multimodal_feat), dim=-1)
        output = self.output_Linear(output)
        return output


class ACN(nn.Module):
    def __init__(self, pretrained_word_matrix):
        super(ACN, self).__init__()
        self.bi_lstm = bi_LSTM(pretrained_word_matrix)
        self.img_Linear = nn.Linear(512, 200)
        self.co_att = Co_Attention()
        self.gmf = GMF()
        self.filtration_gate = FiltrationGate()
        self.crf = CRF(len(TweetProcessor.get_labels()), batch_first=True)

    def forward(self, word_ids, char_ids, img, mask, label_ids):
        text_feature = self.bi_lstm(word_ids, char_ids)
        img_feature = self.img_Linear(torch.tanh(img))
        att_text_features, att_img_features = self.co_att(text_feature, img_feature)
        mult_feature = self.gmf(att_text_features, att_img_features)
        logits = self.filtration_gate(text_feature, mult_feature)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss *= -1

        return loss, logits













