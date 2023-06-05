import numpy as np

import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as F

from scipy.ndimage import binary_dilation, gaussian_filter


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
    """Performs the frankenstein flip on the provided image.
    """

    def __call__(self, image):
        image_array = np.array(image)

        upper, lower = np.split(image_array, 2)
        lower = np.flip(lower, axis=1)
        image_array = np.vstack((upper, lower))

        left, right = np.split(image_array, 2, axis=1)
        right = np.flip(right, axis=0)
        image_array = np.hstack((left, right))

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


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


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
    tfs.Normalize(mean=mean,std=std)])

FOREGROUND = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    DiscardMaskAndBox()
]),
    tfs.ToTensor(),
    tfs.Resize((224, 224), interpolation=tfs.InterpolationMode.NEAREST),
    tfs.ConvertImageDtype(torch.float32),
    tfs.Normalize(mean=mean,std=std)])

SHILOUETTE = tfs.Compose([MyCompose([
    EnlargeImageAndMask(),
    FillOutMask(inverted=True, fill=255),
    FillOutMask(inverted=False, fill=0),
    DiscardMaskAndBox()
]),
    tfs.ToTensor(),
    tfs.Resize((224, 224), interpolation=tfs.InterpolationMode.NEAREST),
    tfs.ConvertImageDtype(torch.float32),
    tfs.Normalize(mean=mean,std=std)])

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
    tfs.Normalize(mean=mean,std=std)])

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
    tfs.Normalize(mean=mean,std=std)])
