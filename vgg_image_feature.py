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

import numpy as np


pre_trained_vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.model = nn.Sequential(*list(pre_trained_vgg16.features))

    def forward(self, x):
        return self.model(x)
#         # block1
#         self.Conv_1 = nn.Conv2d(in_channels=3,
#                                 out_channels=64,
#                                 padding=1,
#                                 kernel_size=(3, 3),
#                                 stride=(1, 1))
#         self.relu_1 = nn.ReLU()
#         self.Conv_2 = nn.Conv2d(in_channels=64,
#                                 out_channels=64,
#                                 padding=1,
#                                 kernel_size=(3, 3),
#                                 stride=(1, 1))
#         self.relu_2 = nn.ReLU()
#         self.Max_pool1 = nn.MaxPool2d(kernel_size=2,
#                                       stride=2)
#
#         # block 2
#         self.Conv_3 = nn.Conv2d(in_channels=64,
#                                 out_channels=128,
#                                 padding=1,
#                                 kernel_size=(3, 3),
#                                 stride=(1, 1))
#         self.relu_3 = nn.ReLU()
#         self.Conv_4 = nn.Conv2d(in_channels=128,
#                                 out_channels=128,
#                                 padding=1,
#                                 kernel_size=(3, 3),
#                                 stride=(1, 1))
#         self.relu_4 = nn.ReLU()
#         self.Max_pool2 = nn.MaxPool2d(kernel_size=2,
#                                       stride=2)
#
#         # block 3
#         self.Conv_5 = nn.Conv2d(in_channels=128,
#                                 out_channels=256,
#                                 padding=1,
#                                 kernel_size=(3, 3),
#                                 stride=(1, 1))
#         self.relu_5 = nn.ReLU()
#         self.Conv_6 = nn.Conv2d(in_channels=256,
#                                 out_channels=256,
#                                 padding=1,
#                                 kernel_size=(3, 3),
#                                 stride=(1, 1))




if __name__ == "__main__":
    dataPath = './data/ner_img/'  # root image path
    mean_pixel = [103.939, 116.779, 123.68]
    tweet_data_path = './data/tweet/all.txt'  # all.txt store
    store_img_feature = './data/img_vgg_feature_224.h5'  # image feature stored file
    mean_pixel = [103.939, 116.779, 123.68]

    img_id_list = []

    # with codecs.open(tweet_data_path, 'r') as file:
    #     for line in file:
    #         rev = line.split('\t')
    #         img_id_list.append(rev[0])

    # for item in img_id_list:
    #     print("process ", item, '.jpg')
    #     img_path = dataPath + item + '.jpg'
    #     print(img_path)
    #     try:
    #         im = cv2.resize(cv2.imread(img_path), (224,224))
    #     except:
    #         continue
    #     for c in range(3):
    #         im[:, :, c] = im[:, :, c] - mean_pixel[c]
    #
    #     im = im.transpose((2, 0, 1))
    #     im = np.expand_dims(im, axis=0)
    #
    #     start = time.time()
    #
    #     features = get_features(model, 30, im)
    #     feat = features[0]
    #
    #     print '%s feature extracted in %f  seconds.' % (img_path, time.time() - start)
    #     vgg_img_feature.create_dataset(name=item, data=feat)


    im = cv2.resize(cv2.imread("./data/ner_img/27.jpg"), (224, 224))
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    im = torch.tensor(im).float()

    model = VGG_16()
    result = model.forward(im)
    feat = result[0]
