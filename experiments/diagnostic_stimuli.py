from datasets.pascal_voc import PascalVoc

import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as F

import torch.nn.functional as TF
import torch.utils.data as data

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from datasets.imagenet_classes import get_imagenet_class_mapping
from models.architecture import FeedForwardTower


class FillOutMask:
    def __init__(self, inverted=False, fill=0):
        self.fill = (fill, fill, fill)
        self.inverted = inverted

    def __call__(self, image, bbox, mask):
        image_array = np.array(image).astype(np.uint8)
        mask_array = np.array(mask).astype(np.uint8)

        if not self.inverted:
            target_color = (255, 255, 255)
        else:
            target_color = (0, 0, 0)

        mask_indices = np.where((mask_array == target_color).all(axis=-1))
        image_array[mask_indices] = self.fill

        return F.to_pil_image(image_array), bbox, mask


class DiscardMaskAndBox():
    def __call__(self, image, bbox, mask):
        return image


class EnlargeAndCropBoundingBox:
    def __init__(self, factor=1.4):
        self.factor = factor

    def __call__(self, image, bbox):
        x, y, w, h = bbox

        new_w = w * self.factor
        new_h = h * self.factor

        new_x = max(0, x - (new_w - w) / 2)
        new_y = max(0, y - (new_h - h) / 2)

        # Make sure the new bounding box is within the image boundaries
        img_width, img_height = image.size
        new_x = min(new_x, img_width - new_w)
        new_y = min(new_y, img_height - new_h)

        # Crop the image
        cropped_image = F.crop(image, int(new_y), int(new_x), int(new_h), int(new_w))

        # Update the bounding box coordinates in the cropped image
        updated_bbox = torch.tensor([x - new_x, y - new_y, w, h], dtype=torch.float32)

        return cropped_image, updated_bbox


class EnlargeImageAndMask():
    def __init__(self, factor=1.4):
        self.enlarge_tfs = EnlargeAndCropBoundingBox(factor=factor)

    def __call__(self, image, bbox, mask):
        new_img, new_bbox = self.enlarge_tfs(image, bbox)
        new_mask, _, = self.enlarge_tfs(mask, bbox)
        return new_img, new_bbox, new_mask


class FrankensteinFlip:
    def __call__(self, image):
        image_array = np.array(image)

        upper, lower = np.split(image_array, 2)
        lower = np.flip(lower, axis=1)
        image_array = np.vstack((upper, lower))

        left, right = np.split(image_array, 2, axis=1)
        right = np.flip(right)
        image_array = np.hstack((left, right))

        return F.to_pil_image(image_array)


class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        state = img
        for t in self.transforms:
            state = t(*state)
        return state


def plot_16_images(ds, title):
    fig = plt.figure(figsize=(40, 40))
    columns = 4
    rows = 4
    for i in range(1, columns * rows + 1):
        img = ds[i][0]
        # bnd = ds[i][0][1]
        # rect = patches.Rectangle((bnd[0], bnd[1]), bnd[2], bnd[3], linewidth=1, edgecolor='r', facecolor='none')
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        # ax.add_patch(rect)
        plt.axis('off')

    plt.title(title)
    plt.show()


tfs_normal = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    DiscardMaskAndBox()
]),
    tfs.Resize((224, 224)),
    tfs.ToTensor()])

tfs_foreground = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    DiscardMaskAndBox()
]),
    tfs.Resize((224, 224)),
    tfs.ToTensor()])

tfs_shilouette = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    DiscardMaskAndBox()
]),
    tfs.Resize((224, 224)),
    tfs.ToTensor()])

tfs_frankenstein = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    DiscardMaskAndBox()
]),
    tfs.Resize((224, 224)),
    FrankensteinFlip(),
    tfs.ToTensor()])


def permutation_test(probs, labels, n=1000):
    results = np.array([np.mean(np.random.permutation(probs) == labels) for _ in range(n)])
    return results


def p_value(random_dist, model_acc):
    return ((random_dist >= model_acc).sum() + 1) / len(random_dist)


def classify(model,
             data_loader,
             imagenet2voc,
             device: str = 'cpu'):
    model.to(device)

    model.eval()
    val_correct = 0
    val_examples = 0
    all_voc_predicitions = torch.empty((0))
    all_labels = torch.empty((0))
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        all_labels = torch.cat((all_labels, targets), dim=0)

        outputs = model(inputs)
        probabilities = TF.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)

        voc_predicted = torch.as_tensor([imagenet2voc[p] for p in predicted])
        voc_predicted = voc_predicted.to('cpu')

        all_voc_predicitions = torch.cat((all_voc_predicitions, voc_predicted), dim=0)

        val_correct += torch.sum(voc_predicted == targets).item()
        val_examples += predicted.shape[0]

    val_accuracy = (val_correct / val_examples)
    print(f'Got {val_accuracy:.2f} accuracy on diagnostic stimuli')

    perm_distributions = permutation_test(all_voc_predicitions, all_labels)

    print(f'Got {np.mean(perm_distributions):.2f} mean permutation accuracy on diagnostic stimuli')

    p_val = p_value(perm_distributions, val_accuracy)
    print(f'Got {p_val:.2f} as p-value on diagnostic stimuli')


if __name__ == '__main__':
    normal_ds = PascalVoc('S:\datasets\pascal_voc', 'trainval', transform=tfs_normal)
    foreground_ds = PascalVoc('S:\datasets\pascal_voc', 'trainval', transform=tfs_foreground)
    shilouette_ds = PascalVoc('S:\datasets\pascal_voc', 'trainval', transform=tfs_shilouette)
    frankenstein_ds = PascalVoc('S:\datasets\pascal_voc', 'trainval', transform=tfs_frankenstein)

    all_datasets = [(normal_ds, 'normal'), (foreground_ds, 'foreground'), (shilouette_ds, 'shilouette'),
                    (frankenstein_ds, 'frankenstein')]

    weights_file = 'rnn-layernorm-ts3.weights'

    _, _, imagenet2voc = get_imagenet_class_mapping(r'S:\datasets\pascal_voc')

    # plot_16_images(foreground_ds, 'Foreground Images')

    model = FeedForwardTower(tower_type='normal', cell_type='rnn', cell_kernel=3, time_steps=3,
                             normalization='layernorm')

    state = torch.load(f'../bw_cluster_weights/{weights_file}')
    model.load_state_dict(state)

    # plot_16_images(shilouette_ds, 'Shilouette Images')

    # plot_16_images(frankenstein_ds, 'Frankenstein Images')

    for ds, ds_name in all_datasets:
        ds_loader = data.DataLoader(ds, batch_size=16)
        print(f'Now processing {ds_name} stimuli')
        classify(model, ds_loader, imagenet2voc, device='cuda')
