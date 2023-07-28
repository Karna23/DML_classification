from torchvision import transforms
import random
from PIL import ImageFilter
import torch
    
class Resizer(object):

    def __init__(self, cfg, inference = False):
        self.output_smallest_size = cfg.input_size
        self.inference = inference

    def __call__(self, sample):
        sample['image'] = transforms.functional.resize(img = sample['image'],
                                                       size = (int(self.output_smallest_size)))
        return sample

class CenterCrop(object):

    def __init__(self, cfg, inference = False):
        
        self.size = cfg.input_size
        self.inference = inference
        
    def __call__(self, sample):
        sample['image'] = transforms.functional.center_crop(img = sample['image'],
                                                            output_size  = (int(self.size)))
        return sample

class PadToSquare(object):

    def __init__(self, cfg, inference = False):
        output_biggest_size = cfg.input_size
        input_width = cfg.width
        input_height = cfg.height
        if input_width<input_height:
            self.width_pad = int((input_height - input_width)/2)
            self.height_pad = 0
        elif input_height<input_width:
            self.height_pad = int((input_width - input_height)/2)
            self.width_pad = 0
        else:
            pass
        self.inference = inference

    def __call__(self, sample):
        sample['image'] = transforms.functional.pad(img = sample['image'],
                                                    padding = (self.width_pad,
                                                               self.height_pad),
                                                    fill = 0,
                                                    padding_mode = 'constant')
        return sample

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        
        if torch.rand(1)[0] < self.p:
            sigma = random.random() * 1.9 + 0.1
            x["image"] = x["image"].filter(ImageFilter.GaussianBlur(sigma))
        else:
            pass
        
        return x

class RandomVerticalFlip(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):

        if torch.rand(1)[0] < self.p:
            sample['image'] = transforms.functional.vflip(sample['image'])
        return sample
    
class RandomHorizontalFlip(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):

        if torch.rand(1)[0] < self.p:
            sample['image'] = transforms.functional.hflip(sample['image'])
        return sample

class ColorJitter(object):
    def __init__(self,
                 brightness=0.4,
                 contrast=0.4,
                 saturation=0.2,
                 hue=0.1, p=0.2):
        
        self.p = p
        self.apply = transforms.ColorJitter(brightness=0.4,
                                            contrast=0.4,
                                            saturation=0.2,
                                            hue=0.1)

    def __call__(self, x):
        
        if random.random() < self.p:
            x["image"] = self.apply(x["image"])
        else:
            pass
        
        return x
    
class RandomGrayscale(object):
    def __init__(self,
                 brightness=0.4,
                 contrast=0.4,
                 saturation=0.2,
                 hue=0.1, p=0.2):
        
        self.p = p
        self.apply = transforms.RandomGrayscale()

    def __call__(self, x):
        
        if random.random() < self.p:
            x["image"] = self.apply(x["image"])
        else:
            pass
        
        return x

class ToTensor(object):
    def __init__(self):
        self.apply = transforms.ToTensor()
    
    def __call__(self, x):
        x["image"] = self.apply(x["image"])
        return x
    
class Normalize(object):
    def __init__(self,mean = [0.485, 0.456, 0.406],
                      std  = [0.229, 0.224, 0.225]):
        self.apply = transforms.Normalize(mean, std)
    
    def __call__(self, x):
        x["image"] = self.apply(x["image"])
        return x

class Identity(object):
    def __call__(self, x):
        return x

class TransformTrain:
    
    # RandomVerticalFlip: Done
    # RandomHorizontalFlip: Done
    # ColorJitter: Done
    # RandomGrayscale: Done
    # GaussianBlur: Done
    # Solarization: Done
    # ToTensor: Done
    # Normalize: Done
    # Identity: Done
    
    def __init__(self, cfg):
        self.transform = transforms.Compose([
            ColorJitter(brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1, p=0.2) if cfg.augmentations.color_jitter else Identity(),
            RandomGrayscale(p=0.1) if cfg.augmentations.random_gray_scale else Identity(),
            GaussianBlur(p=0.1) if cfg.augmentations.gaussian_blur else Identity(),
            PadToSquare(cfg) \
                if cfg.pad_to_square else Identity(),
            Resizer(cfg),
            CenterCrop(cfg),
            ToTensor(),
            RandomVerticalFlip(p=cfg.augmentations.random_vertical_flip) \
                if cfg.augmentations.random_vertical_flip else Identity(),
            RandomHorizontalFlip(p=cfg.augmentations.random_horizontal_flip) \
                if cfg.augmentations.random_horizontal_flip else Identity(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]) \
                if cfg.augmentations.normalize else Identity()
            ])

    def __call__(self, x):
        y = self.transform(x)
        return y
    
class TransformEvaluate:
    def __init__(self, cfg):
        self.transform = transforms.Compose([PadToSquare(cfg) \
                                                 if cfg.pad_to_square else Identity(),
                                             Resizer(cfg),
                                             CenterCrop(cfg),
                                             ToTensor(),
                                             Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225]) if cfg.augmentations.normalize else Identity()])
    def __call__(self, x):
        y = self.transform(x)
        return y
    
class TransformInference:
    def __init__(self, cfg):
        self.transform = transforms.Compose([PadToSquare(cfg, inference=True) \
                                                 if cfg.pad_to_square else Identity(),
                                             Resizer(cfg, inference=True),
                                             CenterCrop(cfg, inference=True),
                                             ToTensor(),
                                             Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225]) if cfg.augmentations.normalize else Identity()])
    def __call__(self, x):
        y = self.transform(x)
        return y
