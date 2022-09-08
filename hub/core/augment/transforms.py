import hub
import cv2 as cv
import numpy as np
from torchvision.transforms import TrivialAugmentWide
import torch




def is_grayscale_image(image):
    """
    Helper function to spot gray_scale images.
    """
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)


class TrivialAugment(TrivialAugmentWide):
  def __init__(self,exclude_transforms=None, include_transforms=None):
    """
    Inherited class from pytorch TrivialAugmentWide. Helps change Augmentation Space. Can either include
    transforms or exclude transforms. There are 14 transforms available from the Wide Augmentation Space(refer TA paper)

    Args:
      exclude_transforms: List of transform_names to excludes
      include_transforms: List of transform_names to include
    """
    super().__init__()
    self.exclude_transforms = exclude_transforms
    self.include_transforms = include_transforms

  def _augmentation_space(self, num_bins: int):
    """
    Returns a changed augmentation space that is a subset of the wide augmentation space.
    """
    aug_space =  {
      # op_name: (magnitudes, signed)
      "Identity": (torch.tensor(0.0), False),
      "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
      "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
      "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
      "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
      "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
      "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
      "Color": (torch.linspace(0.0, 0.99, num_bins), True),
      "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
      "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
      "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
      "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
      "AutoContrast": (torch.tensor(0.0), False),
      "Equalize": (torch.tensor(0.0), False),
    }
    assert not (self.exclude_transforms!=None and self.include_transforms!=None)
    if self.exclude_transforms!=None:
      for transform_name in self.exclude_transforms:
        del aug_space[transform_name]
      return aug_space
    elif self.include_transforms!=None:
      aug_space_additive = {}
      for transform_name in self.include_transforms:
        aug_space_additive[transform_name] = aug_space[transform_name]
      return aug_space_additive
    return aug_space
      
  #Needs to be instantiated only once
@hub.compute
def trivial_augment(image, include_transforms=None, exclude_transforms=None):
  """
  Applies TrivialAugment on a tensor.
  """
  trivial_augmenter = TrivialAugment(include_transforms=include_transforms, exclude_transforms=exclude_transforms)  

  image = torch.from_numpy(image)
  image = image.permute(2, 0, 1)
  image = trivial_augmenter.forward(image).permute(1, 2, 0).numpy()
  
  return image


@hub.compute
def resize(image, height, width, interpolation=cv.INTER_LINEAR, p=1):
  from albumentations import Resize
  resize = Resize(height, width, interpolation=interpolation, p=p)
  return resize.apply(image, interpolation)
  

@hub.compute
def normalize(image, mean, std):
  from albumentations import Normalize
  normalize = Normalize(mean, std)
  return normalize.apply(image)


@hub.compute
def horizontalflip(image):
  flipped = np.ascontiguousarray(image[:, ::-1, ...])
  return flipped


@hub.compute
def verticalflip(image):
  flipped = np.ascontiguousarray(image[::-1, ...])
  return flipped


@hub.compute
def translate(image, dx, dy):
  height, width = image.shape[:2]
  center = (height//2, width//2)
  rot_mat = cv.getRotationMatrix2D( center, 0, 1 )
  rot_mat[0, 2] += dx * width
  rot_mat[1, 2] += dy * height
  final = cv.warpAffine(image, rot_mat, (width, height))
  return final


@hub.compute
def scale_rotate(image, angle: float = 0, scale: float = 1):  
  center = (image.shape[1]//2, image.shape[0]//2)
  rot_mat = cv.getRotationMatrix2D( center, angle, scale )
  final = cv.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))
  return final


@hub.compute
def adjust_brightness_contrast(image, alpha, beta): #pixel = alpha*pixel + beta*mean  can also use convertScaleAbs instead of look up
                                                    #Beta for brightness and alpha for contrast
  lut = np.arange(0, 256).astype("float32")
  if alpha != 1:
    lut *= alpha
  if beta != 0:     
    lut += beta * np.mean(image)

  lut = np.clip(lut, 0, 255).astype("uint8")
  final = cv.LUT(image, lut)
  return final


@hub.compute
def adjust_saturation(img, factor):
  if factor == 1:
    return img

  if is_grayscale_image(img):
    gray = img
    return gray
  else:
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)

  if factor == 0:
    return gray

  result = cv.addWeighted(img, factor, gray, 1 - factor, 0)
  return result


@hub.compute
def posterize(image, bits: int):       #change how many bits to use per channel per pixel
  if np.any((bits < 0) | (bits > 8)):
    raise ValueError("bits must be in range [0, 8]")
  # if not bits.shape or len(bits) == 1:
  if bits == 0:
      return np.zeros_like(image)
  if bits == 8:
      return image.copy()   
  lut = np.arange(0, 256, dtype = np.uint8)
  mask = ~np.uint8(2 ** (8 - bits) - 1)
  lut &= mask
  return cv.LUT(image, lut)


@hub.compute
def solarize(image, threshold):    
  lut = [(i if i < threshold else 255 - i) for i in range(256)]
  prev_shape = image.shape
  image = cv.LUT(image, np.array(lut, dtype="uint8"))

  if len(prev_shape) != len(image.shape):
      image = np.expand_dims(image, -1)
  return image


@hub.compute
def equalize(image):
  if is_grayscale_image(image):
    return cv.equalizeHist(image)
  b,g,r = cv.split(image)
  equ_b = cv.equalizeHist(b)
  equ_g = cv.equalizeHist(g)
  equ_r = cv.equalizeHist(r)
  equalized = cv.merge((equ_b, equ_g, equ_r))
  return equalized


@hub.compute
def invert(image):
  return 255-image


@hub.compute
def shearX(image, shear):
  import math
  shear = np.deg2rad(shear)
  H, W, _ = image.shape
  matx = np.array([
          [1, math.sin(shear), 0],
          [0, math.cos(shear), 0],
  ])
  sheared = cv.warpAffine(image, matx, (W, H))
  return sheared



@hub.compute
def shearY(image, shear):
  import math
  shear = np.deg2rad(shear)
  H, W, _ = image.shape
  maty = np.array([
          [math.cos(shear), 0, 0],
          [math.sin(shear), 1, 0],
  ])
  sheared = cv.warpAffine(image, maty, (W, H))
  return sheared