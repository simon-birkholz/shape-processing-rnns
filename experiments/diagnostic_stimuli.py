from datasets.pascal_voc import PascalVoc

import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class FillOutMask:
    def __init__(self, inverted=False, fill=0):
        self.fill = (fill,fill,fill)
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


tfs_foreground = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    DiscardMaskAndBox()
]),
    tfs.Resize((224,224))])

tfs_shilouette = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    DiscardMaskAndBox()
]),
    tfs.Resize((224,224))])

tfs_frankenstein = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    DiscardMaskAndBox()
]),
    tfs.Resize((224,224)),
    FrankensteinFlip()])

if __name__ == '__main__':
    foreground_ds = PascalVoc('S:\datasets\pascal_voc', 'trainval', transform=tfs_foreground)
    shilouette_ds = PascalVoc('S:\datasets\pascal_voc', 'trainval', transform=tfs_shilouette)
    frankenstein_ds = PascalVoc('S:\datasets\pascal_voc', 'trainval', transform=tfs_frankenstein)

plot_16_images(foreground_ds, 'Foreground Images')

plot_16_images(shilouette_ds, 'Shilouette Images')

plot_16_images(frankenstein_ds, 'Frankenstein Images')
