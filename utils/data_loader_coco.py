import torch
import torch.utils.data as data
import json
import nltk
import cPickle as pickle
import argparse
from build_vocab import Vocabulary


class CocoDataset(data.Dataset):

    def __init__(self, image_root, feature_path, ann_path, vocab_path):
        self._vocab = pickle.load(open(vocab_path))
        self._image_root = image_root
        self._feature_path = feature_path
        self._ann_path = ann_path
        self._annotations = json.load(open(ann_path))
        self._features = torch.load(feature_path)
        self._build_index()
        self._features = self._features[1]

    def _build_index(self):
        imageId_to_image = {}
        imageName_to_image = {}
        imageName_to_feature_idx = {}
        data = self._annotations
        for image in data['images']:
            image_id = image['id']
            image_name = image['file_name']
            imageId_to_image[image_id] = image
            imageName_to_image[image_name] = image

        for idx, name in enumerate(self._features[0]):
            imageName_to_feature_idx[name] = idx

        self._imageName_to_feature_idx = imageName_to_feature_idx
        self._imageId_to_image = imageId_to_image
        self._imageName_to_image = imageName_to_image

    def _get_caption_tensor(self, caption):
        vocab = self._vocab
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        target = list()
        target.append(vocab('<start>'))
        target.extend([vocab(word) for word in tokens])
        target.append(vocab('<end>'))
        target = torch.Tensor(target)
        return target

    def __getitem__(self, index):
        ann = self._annotations['annotations'][index]
        caption = ann['caption']
        image_id = ann['image_id']
        image = self._imageId_to_image[image_id]
        feature_idx = self._imageName_to_feature_idx[image['file_name']]
        feature = self._features[feature_idx] # 512*196
        caption = self._get_caption_tensor(caption)

        return feature, caption

    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        features, captions = zip(*data)

        features = torch.stack(features, 0)
        # batch_size-by-512-196

        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return features, targets, lengths

    def __len__(self):
        return len(self._annotations['annotations'])


def get_loader(image_root, feature_path, ann_path, vocab_path, batch_size=64,
               shuffle=True, num_workers=2):

    coco = CocoDataset(image_root=image_root,
                       feature_path=feature_path,
                       ann_path=ann_path,
                       vocab_path=vocab_path)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=coco.collate_fn)
    return data_loader




