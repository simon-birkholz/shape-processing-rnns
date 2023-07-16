import numpy as np

import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as F

from scipy.ndimage import binary_dilation, gaussian_filter
from datasets.pascal_voc import PascalVoc
import matplotlib.pyplot as plt


class FillOutMask:
    """Takes an image, and a mask and fills out the image where the mask is to the requested color."""

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
    """ Takes a image and a bounding box and crops the images by the provided factor bigger than the bounding box.
        It then adjusts the values of the bounding box and returns the cropped image and the updated bounding box
    """

    def __init__(self, factor=1.4):
        self.factor = factor

    def __call__(self, image, bbox):
        x, y, w, h = bbox

        img_width, img_height = image.size

        # Make sure the new bounding box is not bigger than the original image
        new_w = min(img_width, w * self.factor)
        new_h = min(img_height, h * self.factor)

        new_x = max(0, x - (new_w - w) / 2)
        new_y = max(0, y - (new_h - h) / 2)

        # Make sure the new bounding box is within the image boundaries
        img_width, img_height = image.size
        new_x = min(new_x, img_width - new_w)
        new_y = min(new_y, img_height - new_h)

        new_w = min(new_w, img_width - new_x)
        new_h = min(new_h, img_height - new_y)

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
    """Performs the frankenstein flip on the provided image.
    """

    def __call__(self, image):
        image_array = np.array(image)

        upper, lower = np.split(image_array, 2)

        # taken from https://github.com/cJarvers/shapebias/blob/main/src/mappings.py
        #obj_indices = np.argwhere(lower[0,:,:])
        obj_indices = np.where((lower[0,:,:] == 0).all(axis=-1))
        first = obj_indices[0].min()
        last = obj_indices[0].max()
        shift = first - (lower.shape[1] - last)
        lower = np.flip(lower, axis=1)
        if shift > 0:  # shift right
            lower[:, shift:, :] = lower[:, :-shift, :]
            lower[:, :shift, :] = 255  # fill undefined area with 0
        elif shift < 0:  # shift left
            lower[:, :shift, :] = lower[:, -shift:, :]
            lower[:, shift:, :] = 255  # fill undefined area with 0
        else:  # for shift == 0, do nothing
            pass


        image_array = np.vstack((upper, lower))

        # removed second vertical flip
        #left, right = np.split(image_array, 2, axis=1)
        #right = np.flip(right, axis=0)
        #image_array = np.hstack((left, right))

        return F.to_pil_image(image_array)


class SerratedDilation:
    """Takes an image and a mask. It then uses the mask to determine the borders by a binary dilation and then applies gaussian noise to the border."""

    def __init__(self, borderwidth=10, sigma=2.0):
        self.borderwidth = borderwidth
        self.sigma = sigma

    def __call__(self, image, bbox, mask):
        image_array = np.average(np.array(image), axis=-1)
        mask_array = np.average(np.array(mask), axis=-1)

        x, y, w, h = bbox

        dilated = binary_dilation(mask_array, iterations=self.borderwidth)

        # Taken from https://github.com/cJarvers/shapebias/blob/main/src/mappings.py

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


class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None

    Args:
        image in, image out, nothing is done
    """

    def __call__(self, image):
        return image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

DO_NORM = False

DEV_TEST = tfs.Compose([MyCompose([
    DiscardMaskAndBox()
]),
    tfs.Resize((224, 224)),
    tfs.ToTensor()])

NORMAL = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    DiscardMaskAndBox()
]),
    tfs.ToTensor(),
    tfs.Resize((224, 224), interpolation=tfs.InterpolationMode.NEAREST),
    tfs.ConvertImageDtype(torch.float32),
    tfs.Normalize(mean=mean, std=std) if DO_NORM else NoneTransform()
])

FOREGROUND = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    DiscardMaskAndBox()
]),
    tfs.ToTensor(),
    tfs.Resize((224, 224), interpolation=tfs.InterpolationMode.NEAREST),
    tfs.ConvertImageDtype(torch.float32),
    tfs.Normalize(mean=mean, std=std) if DO_NORM else NoneTransform()
])
SHILOUETTE = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    DiscardMaskAndBox()
]),
    tfs.ToTensor(),
    tfs.Resize((224, 224), interpolation=tfs.InterpolationMode.NEAREST),
    tfs.ConvertImageDtype(torch.float32),
    tfs.Normalize(mean=mean, std=std) if DO_NORM else NoneTransform()
])
FRANKENSTEIN = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    DiscardMaskAndBox()
]),
    tfs.Resize((224, 224), interpolation=tfs.InterpolationMode.NEAREST),
    FrankensteinFlip(),
    tfs.ToTensor(),
    tfs.ConvertImageDtype(torch.float32),
    tfs.Normalize(mean=mean, std=std) if DO_NORM else NoneTransform()
])

SERRATED = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    SerratedDilation(),
    DiscardMaskAndBox()
]),
    tfs.ToTensor(),
    tfs.Resize((224, 224), interpolation=tfs.InterpolationMode.NEAREST),
    tfs.ConvertImageDtype(torch.float32),
    tfs.Normalize(mean=mean, std=std) if DO_NORM else NoneTransform()
])


def avg(a):
    return sum(a) / len(a)


def calculate_mean_and_std(image_ds):
    images = [np.array(img) for img, _ in image_ds]
    means, std_devs = zip(*[(np.mean(img), np.std(img)) for img in images])

    mean = avg(means)
    std_dev = avg(std_devs)
    return mean, std_dev


def plot_images(image_ds, out_file, nof_cols=10, nof_images=100):
    # images = [img for img, _ in image_ds[:nof_images]]
    images = []
    for idx, (img, _) in enumerate(image_ds):
        if idx >= nof_images:
            break
        else:
            images.append(img)

    fig = plt.figure(figsize=(50, 50))
    n_rows = len(images) // nof_cols + 1
    for idx, img in enumerate(images, 1):
        a = fig.add_subplot(n_rows, nof_cols, idx)
        a.imshow(img)
        a.axis('off')

    fig.savefig(out_file)


def check_pascal_voc_images(path_to_dataset):
    non_processed_ds = PascalVoc(path_to_dataset, 'trainval', transform=MyCompose([DiscardMaskAndBox()]))
    cropped_ds = PascalVoc(path_to_dataset, 'trainval',
                           transform=MyCompose([EnlargeImageAndMask(), DiscardMaskAndBox()]))
    foreground_ds = PascalVoc(path_to_dataset, 'trainval',
                              transform=MyCompose(
                                  [EnlargeImageAndMask(),
                                   FillOutMask(inverted=True, fill=255),
                                   DiscardMaskAndBox()]))

    mean, std_dev = calculate_mean_and_std(non_processed_ds)

    plot_images(non_processed_ds, 'non_processed.pdf')

    print(f'Non Processed... Mean: {mean}, Standard Deviation: {std_dev}')

    mean, std_dev = calculate_mean_and_std(cropped_ds)

    plot_images(cropped_ds, 'cropped.pdf')

    print(f'Cropped... Mean: {mean}, Standard Deviation: {std_dev}')

    mean, std_dev = calculate_mean_and_std(foreground_ds)

    plot_images(foreground_ds, 'foreground.pdf')

    print(f'Foreground... Mean: {mean}, Standard Deviation: {std_dev}')


if __name__ == '__main__':
    check_pascal_voc_images('S:\datasets\pascal_voc')
