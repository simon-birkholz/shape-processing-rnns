
from datasets.pascal_voc import PascalVoc

import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches

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



if __name__ == '__main__':
    ds = PascalVoc('S:\datasets\pascal_voc', 'trainval', transform=EnlargeAndCropBoundingBox())

    print(f"Found {len(ds)} images matching constraints")

    w = 10
    h = 10
    fig = plt.figure(figsize=(40, 40))
    columns = 4
    rows = 4
    for i in range(1, columns * rows + 1):
        img = ds[i][0][0]
        bnd = ds[i][0][1]
        rect = patches.Rectangle((bnd[0],bnd[1]), bnd[2], bnd[3], linewidth=1, edgecolor='r', facecolor='none')
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        ax.add_patch(rect)
        plt.axis('off')
    plt.show()