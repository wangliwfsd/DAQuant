import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
import math

__all__ = ['ImageNetDataLoader']


class ImageNetDataLoader(DataLoader):
    def __init__(self, model, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):

        if 'vit' in model:
            # vit
            transform = build_transform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], crop_pct=0.9)
        else:
            # deit
            transform = build_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], crop_pct=0.875)

        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers,
            pin_memory=True)

def build_transform(input_size=224,
                    interpolation='bicubic',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):

    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size,
                interpolation=ip),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

if __name__ == '__main__':
    data_loader = ImageNetDataLoader(
        data_dir='/PATH/TO/ImageNet',
        split='val',
        image_size=384,
        batch_size=16,
        num_workers=0)

    for images, targets in data_loader:
        print(targets)
