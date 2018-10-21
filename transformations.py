import numpy as np
import torch
import cv2
from imgaug import augmenters as iaa
import random
import torchvision.transforms as transforms


def random_crop(image, mask, limit=0.25):
    H_image, W_image = image.shape[:2]

    dy = int(H_image * limit)
    y0 = np.random.randint(0, dy)
    y1 = H_image - np.random.randint(0, dy)

    dx = int(W_image * limit)
    x0 = np.random.randint(0, dx)
    x1 = W_image - np.random.randint(0, dx)

    image = image[y0: y1, x0: x1]
    mask = mask[y0: y1, x0: x1]

    image = cv2.resize(image, (W_image, H_image))
    mask = cv2.resize(mask, (W_image, H_image), interpolation=cv2.INTER_NEAREST)

    mask = mask.reshape((H_image, W_image, 1))

    return image, mask


class ToTensor:
    def __init__(self, predict=False):
        self.predict = predict

    def __call__(self, sample):
        if not self.predict:
            x, y, y_64, y_32, y_16, y_8 = sample['x'], sample['y'],\
                                          sample['y_64'], sample['y_32'], \
                                          sample['y_16'], sample['y_8']

            x = x.transpose((2, 0, 1))
            y = y.transpose((2, 0, 1))
            y_64 = y_64.transpose((2, 0, 1))
            y_32 = y_32.transpose((2, 0, 1))
            y_16 = y_16.transpose((2, 0, 1))
            y_8 = y_8.transpose((2, 0, 1))

            return {'x': torch.from_numpy(x),
                    'y': torch.from_numpy(y),
                    'y_64': torch.from_numpy(y_64),
                    'y_32': torch.from_numpy(y_32),
                    'y_16': torch.from_numpy(y_16),
                    'y_8': torch.from_numpy(y_8),
                    }
        else:
            x = sample['x']
            x = x.transpose((2, 0, 1))
            return {'x': torch.from_numpy(x)}


class Normalize:
    def __init__(self, predict=False):
        self.predict = predict
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        if not self.predict:
            x, y, y_64, y_32, y_16, y_8 = sample['x'], sample['y'],\
                                          sample['y_64'], sample['y_32'], \
                                          sample['y_16'], sample['y_8']
            x = self.normalizer(x)
            return {'x': x,
                    'y': y,
                    'y_64': y_64,
                    'y_32': y_32,
                    'y_16': y_16,
                    'y_8': y_8
                    }
        else:
            x = sample['x']
            x = self.normalizer(x)
            return {'x': x}


class AffineAugmenter:
    def __init__(self):
        self.seq_img = iaa.Sequential([
            iaa.Affine(rotate=(-10, 10), order=1, mode="constant", cval=0, name="MyAffine")
        ])

        self.seq_mask = iaa.Sequential([
            iaa.Affine(rotate=(-10, 10), order=0, mode="constant", cval=0, name="MyAffine")
        ])

    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        seq_img = self.seq_img.localize_random_state()

        seq_img_i = seq_img.to_deterministic()
        seq_masks_i = self.seq_mask.to_deterministic()

        seq_masks_i = seq_masks_i.copy_random_state(seq_img_i, matching="name")

        x = seq_img_i.augment_image(x)
        y = seq_masks_i.augment_image(y)

        return {'x': x,
                'y': y}


class CropAugmenter:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        if random.random() < self.p:
            x, y = random_crop(x, y)

        if random.random() < self.p:
            x = np.array(np.fliplr(x))
            y = np.array(np.fliplr(y))

        return {'x': x,
                'y': y}


class MasksAdder:
    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        y_64 = cv2.resize(y, (64, 64), interpolation=cv2.INTER_NEAREST).reshape(64, 64, 1)
        y_32 = cv2.resize(y, (32, 32), interpolation=cv2.INTER_NEAREST).reshape(32, 32, 1)
        y_16 = cv2.resize(y, (16, 16), interpolation=cv2.INTER_NEAREST).reshape(16, 16, 1)
        y_8 = cv2.resize(y, (8, 8), interpolation=cv2.INTER_NEAREST).reshape(8, 8, 1)

        return {'x': x,
                'y': y,
                'y_64': y_64,
                'y_32': y_32,
                'y_16': y_16,
                'y_8': y_8
                }


class ClassAdder:
    def __call__(self, sample):
        x, y, y_64, y_32, y_16, y_8 = sample['x'], sample['y'], \
                                      sample['y_64'], sample['y_32'], \
                                      sample['y_16'], sample['y_8']

        y_class = torch.sum(y, dim=(1, 2)) > 0
        y_class = y_class.float()

        return {'x': x,
                'y': y,
                'y_64': y_64,
                'y_32': y_32,
                'y_16': y_16,
                'y_8': y_8,
                'y_class': y_class
                }
