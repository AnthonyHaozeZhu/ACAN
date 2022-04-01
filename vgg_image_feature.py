# -*- coding: UTF-8 -*-
"""
@Project ：entity recognition 
@File ：vgg_image_feature.py
@Author ：AnthonyZ
@Date ：2022/3/30 01:02
"""
import torch
import torch.nn as nn
import cv2

import os
import time
import numpy as np


pre_trained_vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)


class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.model = nn.Sequential(*list(pre_trained_vgg16.features))

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    dataPath = './data/ner_img/'  # root image path
    mean_pixel = [103.939, 116.779, 123.68]
    tweet_data_path = './data/tweet/all.txt'  # all.txt store
    store_img_feature = './data/img_vgg_feature_224.h5'  # image feature stored file

    img_id_list = []
    img_features = {}

    for text_filename in ['train', 'dev', 'test']:
        with open(os.path.join(dataPath, text_filename), 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("IMGID:"):
                    img_id_list.append(int(line.replace("IMGID:", "").strip()))

    model = VGG_16()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model.eval()

    for item in img_id_list:
        print("process", item, ".jpg")
        img_path = dataPath + str(item) + '.jpg'
        print(img_path)
        im = cv2.resize(cv2.imread(img_path), (224, 224))
        for c in range(3):
            im[:, :, c] = im[:, :, c] - mean_pixel[c]
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        im = torch.tensor(im).float().to(device)
        start = time.time()
        feature = model.forward(im)
        print('%s feature extracted in %f  seconds.' % (img_path, time.time() - start))
        img_feature = feature.squeeze(0).view(512, 7 * 7)
        img_feature = img_feature.transpose(1, 0)
        img_features[item] = img_feature

    torch.save(img_features, "data/img_vgg_features.pt")
