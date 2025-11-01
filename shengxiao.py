# Copyright (c) 2025 X.J.Wang Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np, numpy.typing as npt
from typing import Any, Literal

from paddle import Tensor
from paddle.io import Dataset
from paddle.vision import transforms
from paddle._typing.dtype_like import _DTypeLiteral

_DatasetMode = Literal['train', 'valid', 'test']

def _transform_image_generator(images):
    mean, std = images.mean(axis=(0, 1, 2)), images.std(axis=(0, 1, 2))
    transformer = transforms.Normalize(mean=mean, std=std, data_format='HWC')
    rotator = transforms.RandomRotation(180)
    for i in range(len(images)):
        normal_image = transformer(images[i])
        rotated_image = rotator(normal_image)
        yield [normal_image, rotated_image]


class SHENGXIAO(Dataset[tuple['_ImageDataType', 'npt.NDArray[Any]']]):
    """
    The `Shengxiao` dataset is the twelve animals image dataset that contains images of Chinese Shengxiao and classification labels.
    It is used for training and testing machine learning models for twelve animals classification recognition.
    Please load this dataset using Python 3.10 or later.

    Args:
        mode (str, optional): The mode of the dataset. It can be 'train', 'valid' or 'test'. Defaults to 'train'.

    Returns:
        tuple: An instance containing the image and the label of Shengxiao dataset.

    """

    NAME = '12-Shengxiao'
    DESCRIPTION = 'The twelve animals image dataset that contains images of Chinese Shengxiao and classification labels.'
    mode: _DatasetMode
    images: list[npt.NDArray[Any]] | list[Tensor]
    labels: list[npt.NDArray[np.int64]] | list[Tensor]
    transform: bool
    dtype: _DTypeLiteral

    def __init__(self, mode: _DatasetMode = 'train', transform: bool = False):
        super().__init__()
        if mode not in ['train', 'valid', 'test']:
            raise ValueError(f"Mode must be 'train', 'valid' or 'test', but got {mode}.")
        self.mode = mode
        self.transform = transform
        self._load_data()

    def _load_data(self):
        data = np.load('shengxiao.npz')
        images, labels = data['images'], data['labels']

        if self.transform:
            transform_data = []
            for idx, img in enumerate(_transform_image_generator(images)):
                transform_data.append(img[0])
                transform_data.append(img[1])

            labels = [val for val in labels for _ in range(2)]
            images, labels = np.array(transform_data), np.array(labels)
            print('Transform is done.')

        train_idx = int(len(images) * 0.8)
        valid_idx = train_idx + int(len(images) * 0.1)

        match self.mode:
            case 'train':
                self.images = images[:train_idx]
                self.labels = labels[:train_idx]
            case 'valid':
                self.images = images[train_idx:valid_idx]
                self.labels = labels[train_idx:valid_idx]
            case 'test':
                self.images = images[valid_idx:]
                self.labels = labels[valid_idx:]

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.images)