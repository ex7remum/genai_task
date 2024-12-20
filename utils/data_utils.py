import os
from PIL import Image
import torch
import numpy as np
from cv2 import resize

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tiff",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    labels = []
    name2idx = {}
    cnt = 0
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
                folder = path.split('/')[-2]
                if folder in name2idx:
                    label = name2idx[folder]
                else:
                    label = cnt
                    name2idx[folder] = cnt
                    cnt += 1
                
                labels.append(label)
                
    return images, labels


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def preprocess_image(image) -> np.ndarray:
    image = np.array(image, dtype=np.uint8)
    image =  np.array(image, dtype=np.float32) / 127.5 - 1
    return torch.tensor(image).permute(2, 0, 1)


def move_batch_to_device(batch: dict, device: torch.device):
    for key, value in batch.items():
        batch[key] = value.to(device)
    return batch
