from datasets.pascal_voc import PascalVoc

import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as F

import torch.nn.functional as TF
import torch.utils.data as data

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse

from typing import List

from datasets.imagenet_classes import get_imagenet_class_mapping
from models.architecture import FeedForwardTower

from scipy.ndimage import binary_dilation, gaussian_filter


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


class SerratedDilation:
    def __init__(self, borderwidth=5, sigma=2.0):
        self.borderwidth = borderwidth
        self.sigma = sigma

    def __call__(self, image, bbox, mask):
        image_array = np.average(np.array(image), axis=-1)
        mask_array = np.average(np.array(mask), axis=-1)

        x, y, w, h = bbox

        dilated = binary_dilation(mask_array, iterations=self.borderwidth)

        noise = np.random.randn(image_array.shape[0], image_array.shape[1])
        noise = (gaussian_filter(noise, sigma=self.sigma) > 0.0) * 255

        result = (255 - mask_array) * (1 - dilated) + dilated * noise
        result = 255 - ((255 - image_array) + (255 - result))
        result = result.astype(np.uint8)
        result = np.stack((result, result, result), axis=2)
        return F.to_pil_image(result), bbox, mask


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
        img = ds[i][0].movedim(0, -1)
        # bnd = ds[i][0][1]
        # rect = patches.Rectangle((bnd[0], bnd[1]), bnd[2], bnd[3], linewidth=1, edgecolor='r', facecolor='none')
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        # ax.add_patch(rect)
        plt.axis('off')

    plt.title(title)
    plt.savefig('example-images.png')


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

tfs_serrated = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    SerratedDilation(),
    DiscardMaskAndBox()
]),
    tfs.Resize((224, 224)),
    tfs.ToTensor()])


def permutation_test(probs, labels, n=1000):
    probs = probs.numpy()
    labels = labels.numpy()
    results = np.array([np.mean(np.random.permutation(probs) == labels, dtype=np.float32) for _ in range(n)])
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
    all_voc_predicitions = torch.empty((0), dtype=torch.int32)
    all_labels = torch.empty((0), dtype=torch.int32)
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        all_labels = torch.cat((all_labels, targets), dim=0)

        outputs = model(inputs)
        probabilities = TF.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)

        voc_predicted = torch.as_tensor([imagenet2voc[p] for p in predicted], dtype=torch.int32)
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

    return {'acc': val_accuracy, 'perm': perm_distributions, 'perm_acc': np.mean(perm_distributions), 'p-value': p_val}


def main(
        datasets: List[str],
        dataset_path: str,
        set: str,
        cell_type: str,
        weights_file: str,
        out_file: str,
        batch_size: int
):
    normal_ds = PascalVoc(dataset_path, set, transform=tfs_normal)
    foreground_ds = PascalVoc(dataset_path, set, transform=tfs_foreground)
    shilouette_ds = PascalVoc(dataset_path, set, transform=tfs_shilouette)
    frankenstein_ds = PascalVoc(dataset_path, set, transform=tfs_frankenstein)
    serrated_ds = PascalVoc(dataset_path, set, transform=tfs_serrated)

    all_datasets = [(normal_ds, 'normal'), (foreground_ds, 'foreground'), (shilouette_ds, 'shilouette'),
                    (frankenstein_ds, 'frankenstein'), (serrated_ds, 'serrated')]

    keep_datasets = [(ds, ds_name) for ds, ds_name in all_datasets if ds_name in datasets]

    _, _, imagenet2voc = get_imagenet_class_mapping(dataset_path)

    # plot_16_images(foreground_ds, 'Foreground Images')

    model = FeedForwardTower(tower_type='normal', cell_type=cell_type, cell_kernel=3, time_steps=3,
                             normalization='layernorm')

    state = torch.load(f'../bw_cluster_weights/{weights_file}')
    model.load_state_dict(state)

    # plot_16_images(shilouette_ds, 'Shilouette Images')

    # plot_16_images(frankenstein_ds, 'Frankenstein Images')

    plot_16_images(serrated_ds, 'Serrated Images')

    accs = {}
    for ds, ds_name in keep_datasets:
        ds_loader = data.DataLoader(ds, batch_size=batch_size)
        print(f'Now processing {ds_name} stimuli')
        accs[ds_name] = classify(model, ds_loader, imagenet2voc, device='cuda')

    plt.figure()
    acc_list = [accs[ds_name]['acc'] for _, ds_name in all_datasets]
    rand_distributions = [accs[ds_name]['perm'] for _, ds_name in all_datasets]
    _, ds_names = zip(*all_datasets)
    x_pos = np.arange(len(all_datasets))
    plt.bar(x_pos - 0.2, acc_list, width=0.4)
    plt.violinplot(rand_distributions, positions=x_pos + 0.2, widths=0.4)

    medians = [np.median(data) for data in rand_distributions]
    plt.scatter(x_pos + 0.2, medians, marker='_', color='tab:blue')
    plt.xticks(x_pos, ds_names)
    plt.savefig(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='Datasets / Diagnostic Stimuli to use as input',
                        nargs='+')
    parser.add_argument('--path', type=str, required=True, help='Path to PascalVOC dataset')
    parser.add_argument('--cell_type', type=str, default='conv', help='Type of (Recurrent) cell to evaluate')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--out', type=str, help='name for the saved diagram', default='output-diagram.png')
    parser.add_argument('--set', type=str, default='trainval', help='PascalVOC image set to use (e.g. train)')
    parser.add_argument('-b', '--batchsize', type=int, default=16)

    args = parser.parse_args()

    main(args.datasets, args.path, args.set, args.cell_type, args.weights, args.out, args.batchsize)
