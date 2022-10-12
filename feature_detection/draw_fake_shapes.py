from typing import Tuple, List, Dict
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.draw import ellipse_perimeter, circle_perimeter, rectangle_perimeter, disk, rectangle, line
from scipy import ndimage
sys.path.append('..')
import utils.constants as constants
import utils.morphological_tools as morph_tools
from PIL import Image
import math
import json 
import cv2

statistics_dict = json.load(open(constants.IMAGESFOLDERPATH.joinpath('statistics.json')))

def generate_ellipsoid(maxLength:int) -> Tuple[List,List]:
    """Generates the coordinates of an ellipsoid in a square

    Args:
        maxLength (int): square dimensions

    Returns:
        Tuple[list,list]: x and y coordinates of the ellipse
    """
    radiusX = random.randint(int(maxLength/4), int(maxLength/3))
    radiusY = random.randint(int(maxLength/4), int(maxLength/3))
    centerX = random.randint(radiusX, maxLength-radiusX )
    centerY = random.randint(radiusY, maxLength-radiusY )
    rr, cc   = ellipse_perimeter(centerX,centerY, radiusX, radiusY)
    return rr, cc
    
def generate_circle(maxLength:int) -> Tuple[list,list]:
    """Generates the coordinates of a circle in a square

    Args:
        maxLength (int): square dimensions

    Returns:
        Tuple[list,list]: x and y coordinates of the circle
    """
    radius = random.randint(int(maxLength/4), int(maxLength/3))
    centerX = random.randint(radius, maxLength-radius )
    centerY = random.randint(radius, maxLength-radius )
    rr, cc   = circle_perimeter(centerX,centerY, radius)
    return rr, cc

def generate_rectangle(maxLength:int) -> Tuple[list,list]:
    """Generates the coordinates of a rectangle in a square

    Args:
        maxLength (int): square dimensions

    Returns:
        Tuple[list,list]: x and y coordinates of the rectangle
    """
    extent_x = random.randint(int(maxLength/4), int(maxLength/3))
    extent_y = random.randint(int(maxLength/4), int(maxLength/3))      
    start_x = random.randint(extent_x, maxLength-extent_x)
    start_y = random.randint(extent_y, maxLength-extent_y)        
    start = (start_x, start_y)
    extent = (extent_x, extent_y)
    rr, cc = rectangle_perimeter(start, extent=extent)
    return rr, cc

def crop(mat:np.uint8, MARGIN:int, sizeImg:int) -> np.uint8 :
    """Center crops a matrix

    Args:
        mat (np.uint8): matrix to crop
        MARGIN (int): number of pixels to crop
        sizeImg (int): size of the image to keep

    Returns:
        np.uint8: cropped matrix
    """
    return mat[MARGIN:MARGIN+sizeImg,MARGIN:MARGIN+sizeImg]

def center_crop(mat, extent) -> np.uint8:
    center_x = int(np.shape(mat)[0]/2)
    center_y = int(np.shape(mat)[1]/2)
    return mat[center_x-int(extent/2): center_x+int(extent/2),center_y-int(extent/2): center_y+int(extent/2)]

def generateStripePattern(sizeImg:int) -> np.uint8:
    """generates a strippe pattern 

    Args:
        sizeImg (int): generates a strippe pattern 

    Returns:
        np.uint8: image to crop
    """
    enclosingSquareLength = int(sizeImg*1.4142)
    mask = np.ones((int(enclosingSquareLength),int(enclosingSquareLength),3), dtype=np.uint8)
    for i in range(constants.DRAWING['stripe_start'], constants.DRAWING['stripe_start']+constants.DRAWING['stripe_width']):
        for j in range(i, enclosingSquareLength, constants.DRAWING['stripe_spacing']):
            mask[j] = 0

    rotationAngle = random.randint(0,90) 
    rotatedImage = ndimage.rotate(mask, rotationAngle, reshape=True,order=1)
    toCrop = np.shape(rotatedImage)[0]-sizeImg
    rotated_mask = morph_tools.normalise(crop(rotatedImage,int(toCrop/2), sizeImg))
    high = 255 * np.ones((int(sizeImg),int(sizeImg),3), dtype=np.uint8) - np.random.poisson(lam=2, size=(sizeImg, sizeImg,3))
    low = np.random.poisson(lam=2, size=(sizeImg, sizeImg,3))
    return np.uint8(rotated_mask*high + (1-rotated_mask)*low)

def generateThickShape(maxRotatedLength:int) -> Tuple[np.uint8, np.uint8]:
    """Draw a synthetic feature

    Args:
        maxRotatedLength (int): Maximum lenght for the rotated shape

    Returns:
        Tuple[np.uint8, np.uint8]: the image, the mask
    """
    shapeVar = random.choices(constants.DRAWING['shapes'], constants.DRAWING['shapes_distribution'])[0]
    if shapeVar == 'rectangle':
        shapeLength = random.randint(int((maxRotatedLength-1)*0.5), maxRotatedLength-1)
        pattern = generateStripePattern(shapeLength)
        mask = np.ones((shapeLength,shapeLength,3),np.uint8)
        mask[0:constants.DRAWING['margin'],:] = 0
        mask[shapeLength-constants.DRAWING['margin']:,:] = 0
        mask[:,0:constants.DRAWING['margin']] = 0
        mask[:,shapeLength-constants.DRAWING['margin']:] = 0
        image = mask*pattern

    else:
        pattern = generateStripePattern(maxRotatedLength)
        mask = np.zeros((maxRotatedLength,maxRotatedLength,3), np.uint8)
        maskBackground = np.ones((maxRotatedLength,maxRotatedLength,3), np.uint8)*255
        rr, cc = disk((int(maxRotatedLength/2), int(maxRotatedLength/2)), math.ceil(maxRotatedLength/2))
        maskBackground[rr,cc] = 0
        rr, cc = disk((int(maxRotatedLength/2), int(maxRotatedLength/2)), math.ceil(maxRotatedLength/2)-constants.DRAWING['margin'])
        mask[rr,cc] = 1
        image = mask*pattern + maskBackground

    # Apply random rotation
    rotationAngle = random.randint(0,180)
    rotatedImage = ndimage.rotate(image, rotationAngle, reshape=True, mode='constant', cval=255,order=1)
    rotatedMaskSegment = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0,order=1)
    return rotatedImage, rotatedMaskSegment*255

def is_cell_available(grid:np.uint8, indexRow:int, indexCol:int) -> bool:
    """Checks if the cell at position (indexRow, indexCol) on the grid is available

    Args:
        grid (np.uint8): the grid 
        indexRow (int): the row index on the surrogate grid
        indexCol (int): the column index on the surrogate grid

    Returns:
        bool: the availability of the cell at position (indexRow, indexCol)
    """
    return not bool(grid[indexRow, indexCol])

def try_square_size(grid:np.uint8, indexRow:int, indexCol:int, gridSize:int) ->int:
    """When randomly partitionning a square into smaller squares, use a surrogate grid (faster) to assess if there is space left to draw in the actual image

    Args:
        grid (np.uint8): the grid 
        indexRow (int): the row index on the surrogate grid
        indexCol (int): the column index on the surrogate grid
        gridSize (int): the grid size

    Returns:
        int: the grid-level size of the shape that can be drawned
    """
    # local copy of the small sizes to consider
    sizes = [1,2,4,8]
    probabilities = list(statistics_dict['shirt_to_probability'].values()).copy()
    
    while True:
        
        potentialSize = random.choices(sizes, probabilities )[0]
        # assert we are not out of the grid
        if indexRow + potentialSize <= gridSize and indexCol + potentialSize <= gridSize:
            # assert the grid is empty for the shape we want to draw
            if np.count_nonzero(grid[indexRow:indexRow+potentialSize, indexCol:indexCol+potentialSize])==0:
                return potentialSize
        else:
            probabilities.pop(sizes.index(potentialSize)) 
            sizes.remove(potentialSize)
            

def generateBlockOfFlats(sizeImg:int) -> Tuple[np.uint8, np.uint8, np.uint8]:
    """Generates a block of houses to replicate semi-detached houses

    Args:
        sizeImg (int): Size of the image

    Returns:
        Tuple[np.uint8, np.uint8, np.uint8]: (the image, the mask and the band (=the surface occupied if the house were not detached))
    """
    # declare the maximum enclosing square that can be drawn in the image considered
    enclosingSquareLength = int(sizeImg*math.sqrt(2))
    # declare the mask and the band
    mask = np.zeros((enclosingSquareLength,enclosingSquareLength,3), dtype=np.uint8)
    band = np.ones((enclosingSquareLength,enclosingSquareLength,3), dtype=np.uint8)*255
    #declare the middle of the images to locate the block of flats in the center of the image
    middle = int(enclosingSquareLength/2)
    #randomly select width
    height = random.choice([int(sizeImg/8),int(sizeImg/4),int(sizeImg/3)])
    start_index = constants.DRAWING['margin']
    reached = False
    while not reached:
        length = random.choice([50,100,150])
        start = (start_index, middle-int(height/2))
        if enclosingSquareLength <= start_index + length:
            extent = (enclosingSquareLength-start_index, height)
            reached = True
        else:            
            extent = (length, height)
        rr, cc = rectangle(start, extent=extent)
        mask[rr,cc] = 1
        start_index += constants.DRAWING['margin'] + length
    band[:,middle-int(height/2)-constants.DRAWING['margin']:middle+int(height/2)+constants.DRAWING['margin']] = 0
    rotationAngle = random.randint(0,180)
    # apply random rotation
    #rotatedBand = ndimage.rotate(band, rotationAngle, reshape=True, mode='constant', cval=255,order=1)
    rotatedImage = ndimage.rotate(band+mask*generateStripePattern(enclosingSquareLength), rotationAngle, reshape=True, mode='constant', cval=255,order=1)
    rotatedMask = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0,order=1)*255
    cropMargin = int((enclosingSquareLength-sizeImg)/2)
    #return crop(rotatedImage, cropMargin, sizeImg), crop(rotatedMask, cropMargin, sizeImg), crop(rotatedBand, cropMargin, sizeImg)
    return crop(rotatedImage, cropMargin, sizeImg), crop(rotatedMask, cropMargin, sizeImg)

def generateFeature(thumbnailSize:int) -> Tuple[np.uint8, np.uint8, str]:  
    size_key = constants.DRAWING['size_to_shirt'][thumbnailSize]
    feature_name = random.choices(statistics_dict['shirt_to_feature_probability'][size_key]['features'], statistics_dict['shirt_to_feature_probability'][size_key]['probabilities'])[0]
    high_level_feature = constants.LOWTOHIGHTDISPLAY[feature_name]
    shape_index = random.randint(0,statistics_dict[high_level_feature][feature_name][size_key]-1)
    shape_path  = constants.IMAGESFOLDERPATH.joinpath(f'{high_level_feature}/{feature_name}/{size_key}/{shape_index}{constants.FILEEXTENSION}')
    mask_path   = constants.IMAGESFOLDERPATH.joinpath(f'{high_level_feature}/{feature_name}/{size_key}/{shape_index}_mask{constants.FILEEXTENSION}')
    image = np.asarray(Image.open(shape_path))
    mask = np.asarray(Image.open(mask_path))
    rotationAngle = random.randint(0,180)
    rotatedImage = ndimage.rotate(cv2.bitwise_or(image, 255-mask), rotationAngle, reshape=True, mode='constant', cval=255, order=1)
    rotatedMask = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0, order=1)
    return rotatedImage, rotatedMask, high_level_feature

def fillThumbnail(thumbnailSize:int, pattern:np.uint8, mask:np.uint8, boundRowLow:int, boundRowHigh:int, boundColLow:int, boundColHigh:int, imageToFill:np.uint8, maskToFill:np.uint8)-> Tuple[np.uint8, np.uint8]:
    """given a pattern, randomly places it in a thumbnail

    Args:
        thumbnailSize (int): Size of the thumbnail to draw
        pattern (np.uint8): pattern to draw
        mask (np.uint8): _description_
        boundRowLow (int): lower row bound of the thumbnail in the image
        boundRowHigh (int): higher row bound of the thumbnail in the image
        boundColLow (int): lower col bound of the thumbnail in the image
        boundColHigh (int): higher col bound of the thumbnail in the image
        imageToFill (np.uint8): image to complete
        maskToFill (np.uint8): mask to complete

    Returns:
        Tuple[np.uint8, np.uint8]: image, associated mask
    """
    # Once we get the pattern, we draw it at a random position in the thumbnail
    thumbnail = np.ones((thumbnailSize,thumbnailSize,3), np.uint8)*255
    maskToReturn =  np.zeros((thumbnailSize,thumbnailSize,3), np.uint8)
    posX = random.randint(0, thumbnailSize-np.shape(pattern)[0])
    posY = random.randint(0, thumbnailSize-np.shape(pattern)[1])
    thumbnail[posX:posX+np.shape(pattern)[0], posY:posY+np.shape(pattern)[1]] = cv2.bitwise_and(thumbnail[posX:posX+np.shape(pattern)[0], posY:posY+np.shape(pattern)[1]], pattern)
    imageToFill[boundRowLow:boundRowHigh, boundColLow:boundColHigh] = thumbnail
    maskToReturn[posX:posX+np.shape(mask)[0], posY:posY+np.shape(mask)[1]] += mask
    maskToFill[boundRowLow:boundRowHigh, boundColLow:boundColHigh] += cv2.cvtColor(maskToReturn, cv2.COLOR_RGB2GRAY)
    return imageToFill, maskToFill

def generateFeatureOrStripe(thumbnailSize:int, boundRowLow:int, boundRowHigh:int, boundColLow:int, boundColHigh:int, image:np.uint8, full_mask:dict, pReal=0.8):
    """Draws a shape on a thumbnail (sub-part) of the image

    Args:
        thumbnailSize (int): Size of the thumbnail to draw
        boundRowLow (int): lower row bound of the thumbnail in the image
        boundRowHigh (int): higher row bound of the thumbnail in the image
        boundColLow (int): lower col bound of the thumbnail in the image
        boundColHigh (int): higher col bound of the thumbnail in the image
        image (np.uint8): Image on which to draw
        masksDict (dict): dictionnary of associated masks (one per feature)
        pReal (float, optional): Probability to draw a real feature (vs a synthetic one). Defaults to 0.8.

    Returns:
        Tuple[np.uint8, Dict]: image, dictionnary of associated masks (one per feature)
    """
    # randomly choose the pattern, real or synthetic
    patternVar = random.choices([0,1], [pReal,1-pReal])[0]
    if patternVar ==0:
        pattern, mask, choice = generateFeature(thumbnailSize)

    else:
        choice = 'imprint'
        if thumbnailSize == 512:
            pattern, mask = generateBlockOfFlats(thumbnailSize)
        else:
            pattern, mask = generateThickShape(int(thumbnailSize/math.sqrt(2)))

    image, mask = fillThumbnail(thumbnailSize, pattern, mask, boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, full_mask[:,:,1+constants.HIGHLEVELFEATURES.index(choice)])
    
    full_mask[:,:,1+constants.HIGHLEVELFEATURES.index(choice)] = mask
    
    return image, full_mask

def addLines(image:np.uint8, sizeImg=512) ->  np.uint8:
    """Adds random lines to the final image

    Args:
        image (np.uint8): image without noise
        sizeImg (int, optional):  Size of the image. Defaults to 512. don't mess around with that

    Returns:
        np.uint8: image + noise
    """
    # Declare the lines image: lines have value 0, not 1.
    lines = np.ones((sizeImg,sizeImg,3), dtype=np.uint8)
    for i in range(random.randint(0,10)):
        if random.randint(0,1) == 0:
            r0, r1 = 0, sizeImg-1
            c0, c1 = random.randint(0,sizeImg-1), random.randint(0,sizeImg-1)
        else:
            c0, c1 = 0,sizeImg-1
            r0, r1 = random.randint(0,sizeImg-1), random.randint(0,sizeImg-1)
        rr, cc = line(r0,c0,r1,c1)
        lines[rr, cc] = 0

    # Once lines are drawn, simple element-wise product of the two matrices
    return lines*image

def generateFeaturesAndMask(sizeImg=512, minSize = 64) -> Tuple[np.uint8, Dict] :
    """Generates one image and the associated masks

    Args:
        patternsDict (Dict):  A dictionnary containing all the features of all the sizes considered (not the shapes as objects but the meta information) 
        sizeImg (int, optional): Size of the image. Defaults to 512. don't mess around with that
        minSize (int, optional): Minimum shape size to consider. Defaults to 32.

    Returns:
        Tuple[np.uint8, Dict]: image, dictionnary of associated masks (one per feature)
    """
    image = np.ones((sizeImg,sizeImg,3), np.uint8)*255
    mask  = np.zeros((sizeImg,sizeImg,4), np.uint8)
    gridSize = int(sizeImg/minSize)
    grid = np.zeros((gridSize,gridSize), np.uint8)
    for indexRow in range(gridSize):
        for indexCol in range(gridSize):
            if is_cell_available(grid, indexRow, indexCol):
                blockSize = try_square_size(grid, indexRow, indexCol, gridSize)
                image, mask  = generateFeatureOrStripe(blockSize*minSize, indexRow*minSize, (indexRow+blockSize)*minSize , indexCol*minSize, (indexCol+blockSize)*minSize, image, mask)
                grid[indexRow:indexRow+blockSize, indexCol:indexCol+blockSize] = 1

    image = addLines(image)
    return image, mask

def make_batch(batchSize:int, n_input_channels:int) ->Tuple[np.uint8, Dict]:
    # Create empty batch for the image
    batch = np.zeros((batchSize, n_input_channels, 512, 512), np.uint8)
    # Create a tensor for the target
    batch_mask = np.zeros((batchSize, 1+len(constants.HIGHLEVELFEATURES), 512, 512), np.uint8)
    # Populate the image batch and the mask batches with the appropriate data
    for batch_index in range(batchSize):
        
        background = np.zeros((512, 512), np.uint8)
        image, masks = generateFeaturesAndMask()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if n_input_channels == 1:
            batch[batch_index,0] = image
        elif n_input_channels == 3:
            batch[batch_index]   = np.concatenate((np.expand_dims(morph_tools.erosion(image),0), np.expand_dims(image,0), np.expand_dims(morph_tools.dilation(image),0)), axis = 0)
        
        _, binarised_masks  = cv2.threshold(masks, 10 ,1, cv2.THRESH_BINARY)

        for feature_index in range(len(constants.HIGHLEVELFEATURES)):     
            batch_mask[batch_index, 1+feature_index] = binarised_masks[:,:,1+feature_index]
            background = cv2.bitwise_or(background,  binarised_masks[:,:,1+feature_index])

        batch_mask[batch_index,0] = 1 - background

    return batch, batch_mask

def main():
    for i in range(20):
        batch, b_m = make_batch(2, 3)
        plt.matshow(batch[0,0])
        plt.show()
        for j in range(b_m.shape[1]):
            plt.matshow(b_m[0,j])
            plt.show()

if __name__ =='__main__':
    main()