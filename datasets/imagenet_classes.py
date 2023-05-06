import os
import re

from nltk.corpus import wordnet as wn
import numpy as np


def get_pascal_voc_class_list(path):
    class_list = []
    with open(os.path.join(path, "pascalvoc_class_list.txt")) as f:
        for line in f:
            label = line.split(' ')[1]
            class_list.append(label.strip())

    return class_list


def get_imagenet_class_list(path):
    class_list = []
    with open(os.path.join(path, "imagenet_categories.txt")) as f:
        for line in f:
            id, label = line.split(' ')[0], ' '.join(line.split(' ')[1:])
            class_list.append((id, label.strip()))

    return class_list


# taken from https://github.com/cJarvers/shapebias/blob/main/src/helpers/imagenet_synsets.py

TRANSLATE = {'diningtable': {'alias': 'dining_table', 'index': 0},
             'pottedplant': {'alias': 'flowerpot', 'index': 0},
             'tvmonitor': {'alias': 'television', 'index': 1},
             'cow': {'alias': 'bovid', 'index': 0},
             'aeroplane': {'alias': 'heavier-than-air_craft', 'index': 0},
             'train': {'alias': 'locomotive', 'index': 0},
             }


def get_imagenet_class_mapping(path):
    imagenet_classes = get_imagenet_class_list(path)
    pascal_voc_classes = get_pascal_voc_class_list(path)

    pascal_synsets = {}
    for label in pascal_voc_classes:
        if label in TRANSLATE.keys():
            synset = wn.synsets(TRANSLATE[label]['alias'], pos='n')[TRANSLATE[label]['index']]
        else:
            synset = wn.synsets(label, pos='n')[0]
        pascal_synsets[label] = synset

    imagenet_synsets = [wn.synset_from_pos_and_offset('n', int(id[1:])) for id, label in imagenet_classes]

    voc2imgnet = {c: [] for c in pascal_voc_classes}
    imgnet2voc = {}
    imagenet2voc = np.zeros(1000, dtype=np.int64)
    for i, synset in enumerate(imagenet_synsets):
        for vocclass, vocsynset in pascal_synsets.items():
            if vocsynset in synset.lowest_common_hypernyms(vocsynset):
                voc2imgnet[vocclass].append(i)
                imgnet2voc[i] = vocclass
                imagenet2voc[i] = pascal_voc_classes.index(vocclass) + 1

    return voc2imgnet, imgnet2voc, imagenet2voc

if __name__ == '__main__':
    voc2imgnet, imgnet2voc, imagenet2voc = get_imagenet_class_mapping(r'S:\datasets\pascal_voc')

    _, imgnet_class_list = zip(*get_imagenet_class_list(r'S:\datasets\pascal_voc'))

    print(f'Could match {len(imgnet2voc)} of {len(imgnet_class_list)}. The following: ')
    for k, v in imgnet2voc.items():
        print(f'{k}:{imgnet_class_list[k]}: {v}')

    print(f'Could not match: ')
    for i in range(0,1000):
        if i not in imgnet2voc.keys():
            print(f'{i}:{imgnet_class_list[i]}')


