import numpy as np
import gc
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader


def load_image(path, mask=False, to256=False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape

    if to256:
        height *= 2
        width *= 2

    # Padding in needed for UNet models because they need image size to be divisible by 32
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    if to256:
        if mask:
            img = cv2.resize(img, (202, 202), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (202, 202), interpolation=cv2.INTER_LINEAR)

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    if mask:
        img = img[:, :, 0:1] // 255
    else:
        img = img / 255.0

    return img.astype(np.float32)


def load_train_data(train_images_path, train_masks_path, clean_small_masks=True, to256=False):
    file_paths = os.listdir(train_images_path)

    images = [load_image(train_images_path + file_path, to256=to256) for file_path in file_paths]
    masks = [load_image(train_masks_path + file_path, mask=True, to256=to256) for file_path in file_paths]

    filtered_images = images
    filtered_masks = masks

    images = np.array(filtered_images)
    masks = np.array(filtered_masks)

    return images, masks


def load_test_data(test_images_path, load_images=False, to256=False):
    file_paths = os.listdir(test_images_path)
    if load_images:
        images = np.array([load_image(test_images_path + file_path, to256=to256) for file_path in file_paths])
        return file_paths, images
    else:
        return file_paths, None


class SaltDataset(Dataset):
    def __init__(self, x, y=None, transform=None, predict=False):
        self.x = x
        self.y = y
        self.predict = predict
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if not self.predict:
            x_ = self.x[idx]
            y_ = self.y[idx]
            result = {'x': x_, 'y': y_}
        else:
            x_ = self.x[idx]
            result = {'x': x_}

        if self.transform:
            result = self.transform(result)

        return result


def build_data_loader(x, y, transform, batch_size, shuffle, num_workers, predict):
    dataset = SaltDataset(x, y, transform=transform, predict=predict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def trim_masks(masks, height, width):

    if masks.shape[1:3] == (256, 256):
        new_masks = []
        for i in range(masks.shape[0]):
            new_mask = masks[i][26: 256 - 26, 26: 256 - 26]
            new_mask = cv2.resize(new_mask, (101, 101), interpolation=cv2.INTER_NEAREST)
            new_mask = new_mask.reshape(101, 101, 1)
            new_masks.append(new_mask)
        masks = np.array(new_masks)
        gc.collect()
    else:
        if height % 32 == 0:
            y_min_pad = 0
            y_max_pad = 0
        else:
            y_pad = 32 - height % 32
            y_min_pad = int(y_pad / 2)
            y_max_pad = y_pad - y_min_pad

        if width % 32 == 0:
            x_min_pad = 0
            x_max_pad = 0
        else:
            x_pad = 32 - width % 32
            x_min_pad = int(x_pad / 2)
            x_max_pad = x_pad - x_min_pad

        masks = masks[:, y_min_pad: 128 - y_max_pad, x_min_pad: 128 - x_max_pad]

    return masks


def load_model(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    return model


def average_weigths(target, sources):
    sources_named_parameters = [dict(source.named_parameters()) for source in sources]
    frac = 1 / len(sources)

    for name, param in target.named_parameters():
        source_params = [frac * source_named_parameters[name].data for source_named_parameters
                         in sources_named_parameters]

        param.data.copy_(sum(source_params))

    return target


def make_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
