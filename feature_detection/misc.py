## Load the mask
## Load the original image
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append('..')
from utils import constants
from utils import morphological_tools as morph_tools

city_name = 'Barrow-in-Furness'
tile_name = '1905021080162'

image_path = constants.CITIESFOLDERPATH.joinpath(f'{city_name}/{tile_name}.jpg')
mask_path  = constants.PROCESSEDPATH.joinpath(f'{city_name}/{tile_name}/background.jpg')

img  = cv2.imread(str(image_path))[:,:,0]

mask = cv2.imread(str(mask_path))
mask = np.transpose(mask,(1,0,2))[:,:,0]

#dst  = cv2.bitwise_and(img, mask) + (255-mask)
dst  = cv2.bitwise_or(cv2.bitwise_and(img, mask),morph_tools.dilation(255-mask))
plt.matshow(dst)
plt.show()

cv2.imwrite('inpainted.jpg', dst)
