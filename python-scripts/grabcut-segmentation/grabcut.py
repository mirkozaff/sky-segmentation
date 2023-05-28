import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt


def main(args):
    # Input image
    img = cv.imread(args.input_image)
    assert img is not None, "file could not be read, check with os.path.exists()"
    mask_refined = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # Draw the rectangle containing all the object of interest
    # Everything outside is marked ad background
    h, w, _ = img.shape
    rect = (0, 0, w, h//3)
    cv.grabCut(img,mask_refined,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

    # First segmentation based on the rectangle area
    mask_coarse = np.where((mask_refined==2)|(mask_refined==0),0,1).astype('uint8')
    img_masked = img*mask_coarse[:,:,np.newaxis]
    plt.imshow(img_masked[:,:,::-1]),plt.colorbar(),plt.show()

    # newmask is the manually labelled mask image
    newmask = cv.imread(args.mask, cv.IMREAD_GRAYSCALE)
    assert newmask is not None, "file could not be read, check with os.path.exists()"

    # Wherever it is marked white (sure foreground), change mask=1
    # Wherever it is marked black (sure background), change mask=0
    # Wherever it is marked gray (ignore pixel)
    mask_refined[newmask == 0] = 0
    mask_refined[newmask == 255] = 1

    # Add information to grabCut
    mask_refined, bgdModel, fgdModel = cv.grabCut(img,mask_refined,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

    # Refined image
    mask_refined = np.where((mask_refined==2)|(mask_refined==0),0,1).astype('uint8')
    img_masked = img_masked*mask_refined[:,:,np.newaxis]
    plt.imshow(img_masked[:,:,::-1]),plt.colorbar(),plt.show()

    # Plotting results
    f, axarr = plt.subplots(2,2)
    axarr[0,0].plot()
    axarr[0, 0].set_title("Original image")
    axarr[0,0].imshow(img[:,:,::-1])
    axarr[0,1].plot()
    axarr[0,1].set_title("Maksked image")
    axarr[0,1].imshow(img_masked[:,:,::-1])
    axarr[1,0].plot()
    axarr[1,0].set_title("Coarse mask")
    axarr[1,0].imshow(mask_coarse)
    axarr[1,1].plot()
    axarr[1,1].set_title("Refined mask")
    axarr[1,1].imshow(mask_refined)
    plt.show(block=True)

    # Save binary mask
    plt.imsave(f'{args.input_image}_binary.jpg', mask_refined, cmap='binary')

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='GrubCut algorithm')
    parser.add_argument('-i', '--input_image', default='Immagine10.jpg',type=str,
                        help='Path to the input image (default: Immagine10.jpg)')
    parser.add_argument('-m', '--mask', default='Immagine10_mask.jpg', type=str,
                        help='Path to the input mask image (default: Immagine10_mask.jpg)')
    args = parser.parse_args()

    main(args)
    