import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class RandomPatchTransform:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)

        # Get dimensions
        c, h, w = img.shape

        # Calculate random crop parameters
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)

        # Perform random crop to get 256x256 patch
        img = TF.crop(img, top, left, self.size, self.size)

        # Random rotation by 90 degree increments
        k = random.randint(0, 3)  # 0=0°, 1=90°, 2=180°, 3=270°
        img = torch.rot90(img, k, dims=[-2, -1])

        # Random flips
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-2])  # Horizontal flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-1])  # Vertical flip

        return img

class HDRTransform:
    def __init__(self, patch_size=256):
        self.size = patch_size
        
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        # Random crop
        c, h, w = img.shape
        # Use static seed for consistent crops across images
        if not hasattr(self, 'i') or not hasattr(self, 'j'):
            self.i = torch.randint(0, h - self.size + 1, (1,)).item()
            self.j = torch.randint(0, w - self.size + 1, (1,)).item()
        img = img[:, self.i:self.i+self.size, self.j:self.j+self.size]
        return img
