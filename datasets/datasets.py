from PIL import Image
from torch.utils.data import Dataset
from utils.data_utils import make_dataset
from torchvision.transforms import v2
from utils.class_registry import ClassRegistry


datasets_registry = ClassRegistry()


@datasets_registry.add_to_registry(name="base_dataset")
class BaseDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.paths, self.labels = make_dataset(root)
        self.transforms = transforms
        self.flip = v2.RandomHorizontalFlip(p=0.5)
        self.classes = max(self.labels) + 1

    def __getitem__(self, ind):
        path = self.paths[ind]
        image = Image.open(path).convert("RGB")
        label = self.labels[ind]

        if self.transforms:
            image = self.transforms(image)
        image = self.flip(image)

        return {"images": image,
                "labels": label}

    def __len__(self):
        return len(self.paths)
    
    def get_num_classes(self):
        return self.classes
