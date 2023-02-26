from datasets.imagenet import get_imagenet


if __name__ == '__main__':
    ds = get_imagenet('H:\datasets\imagenet')
    ds[0][0].show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
