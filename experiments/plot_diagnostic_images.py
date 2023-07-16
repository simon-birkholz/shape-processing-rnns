import matplotlib.pyplot as plt
import argparse

from datasets.pascal_voc import PascalVoc
from ds_transforms import NORMAL, FOREGROUND, SHILOUETTE, FRANKENSTEIN, SERRATED


def plot_images(datasets, out_file, nof_images=4):
    fig = plt.figure(constrained_layout=True)
    rows = len(datasets)
    subfigs = fig.subfigures(nrows=rows, ncols=1)
    for idx in range(0, rows):
        subfigs[idx].suptitle(datasets[idx][1])

        axs = subfigs[idx].subplots(nrows=1, ncols=nof_images)
        for i, ax in enumerate(axs):
            img = datasets[idx][0][i][0].movedim(0, -1)
            ax.imshow(img)
            ax.axis('off')

    fig.savefig(out_file)


def main(dataset_path: str, out_path: str):
    set = 'trainval'
    normal_ds = PascalVoc(dataset_path, set, transform=NORMAL)
    foreground_ds = PascalVoc(dataset_path, set, transform=FOREGROUND)
    shilouette_ds = PascalVoc(dataset_path, set, transform=SHILOUETTE)
    frankenstein_ds = PascalVoc(dataset_path, set, transform=FRANKENSTEIN)
    serrated_ds = PascalVoc(dataset_path, set, transform=SERRATED)

    all_datasets = [(normal_ds, 'normal'), (foreground_ds, 'foreground'), (shilouette_ds, 'shilouette'),
                    (frankenstein_ds, 'frankenstein'), (serrated_ds, 'serrated')]

    plot_images(all_datasets, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to PascalVOC dataset')
    parser.add_argument('--out', type=str, help='name for the saved diagram', default='output-diagram.pdf')

    args = parser.parse_args()

    main(args.path, args.out)
