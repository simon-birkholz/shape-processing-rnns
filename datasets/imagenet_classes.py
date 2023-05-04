import os
import re

from nltk.corpus import wordnet as wn


def get_pascal_voc_class_list():
    return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']


def get_imagenet_class_list(path):
    class_list = []
    with open(os.path.join(path, "imagenet_class_list.txt")) as f:
        for line in f:
            _, label = line.split('\t')
            class_list.append(label.strip())

    return class_list

def get_imagenet_class_mapping():
    imagenet_classes = get_imagenet_class_list('S:\\datasets\\pascal_voc')
    pascal_voc_classes = get_pascal_voc_class_list()

    resulting_mapping = dict()

    for c in pascal_voc_classes:
        syns = wn.synsets(c)
        for s in wn.synsets(c):
            syns += s.hypernyms()
            syns += s.hyponyms()
        syns_name = [s.name().split('.')[0] for s in syns]
        for ic in imagenet_classes:
            for ss in syns_name:
                if re.match(ss, ic):
                    resulting_mapping[ic] = c
                    break

    print(f'Could match {len(resulting_mapping)} of {len(imagenet_classes)} classes. The following:')
    for k,v in resulting_mapping.items():
        print(f'{k}: {v}')

    print(f'Could not match: ')
    for ic in imagenet_classes:
        if ic not in resulting_mapping.keys():
            print(f'{ic}')
    return resulting_mapping



if __name__ == '__main__':
    get_imagenet_class_mapping()
