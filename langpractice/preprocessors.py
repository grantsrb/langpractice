import torch
import numpy as np
import torchvision.transforms as T

def normalize_prep(pic):
    new_pic = 3*(pic - 255/2)/(255/2)
    return new_pic[None]

def null_prep(pic):
    return pic[None]

def pong_prep(pic):
    pic = pic[35:195] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic[pic == 144] = 0 # erase background (background type 1)
    pic[pic == 109] = 0 # erase background (background type 2)
    pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
    return pic[None]

def snake_prep(pic):
    new_pic = np.zeros(pic.shape[:2],dtype=np.float32)
    new_pic[:,:][pic[:,:,0]==1] = 1
    new_pic[:,:][pic[:,:,0]==255] = 1.5
    new_pic[:,:][pic[:,:,1]==255] = 0
    new_pic[:,:][pic[:,:,2]==255] = .33
    pic = new_pic
    return new_pic[None]

color_jitter = T.ColorJitter(
    hue=.3,
    saturation=.5
    #brightness=.1
    #contrast=.1
)
gauss_blur = T.RandomApply([T.GaussianBlur((5,5),sigma=0.1)],p=1)
distort = T.RandomPerspective(distortion_scale=.6, p=0.6)
invert_color = T.RandomInvert()
# trans = (horizontal shift, vertical shift)
affine = T.RandomAffine(180, translate=(.03,0), scale=(.5,1))
pipeline = torch.nn.Sequential(
    color_jitter,
    gauss_blur,
    affine,
    distort,
    invert_color,
)
def sample_augmentation(imgs):
    """
    Samples an augmentation on the argued image(s).

    Args:
        imgs: torch tensor (..., C, H, W)
    """
    return pipeline(imgs)
