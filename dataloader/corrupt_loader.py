import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from corruptions import *
from typing import Callable, Optional, Tuple, Any

class CorruptDataset(ImageFolder):
    def __init__(self, root: str, corruption: str, intensity:int, transform: Optional[Callable] = None):
        super(CorruptDataset, self).__init__(root = root, transform = transform)
        self.corruption = corruption
        self.intensity = intensity
        self.transform = transform
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, _ = self.samples[index]
        img = Image.open(img)
        img = img.convert('RGB')
        tnsr_img = torchvision.transforms.ToTensor()(img)

        if self.corruption and self.intensity is not None:
            if type(self.corruption) == list and type(self.intensity) == list:
                ret = [tnsr_img]
                corr_intensity_lst = zip(self.corruption, self.intensity)

                for corruption, intensity in corr_intensity_lst:
                    ret.append(torch.tensor(key2deg[corruption](tnsr_img, intensity)))

                return tuple(ret) 
            else:
                return tnsr_img, torch.tensor(key2deg[self.corruption](tnsr_img, self.intensity))
        else:
            return tnsr_img