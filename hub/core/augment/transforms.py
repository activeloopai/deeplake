import hub
import cv2 as cv
import numpy as np




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