import argparse
import matplotlib.pyplot as plt
from skimage import segmentation
from grubcut import grabCut
from rfsegmentation import randomForestSegmentation
import os
from os import listdir

def main(args):
    # Read images in input folder
    images = listdir(os.path.join(args.input_dir))
    for im in images:
        # Load image
        img = plt.imread(os.path.join(args.input_dir, im))
        assert img is not None, "file could not be read, check with os.path.exists()"

        mask_coarse = grabCut(img) 
        # Convert the mask labels for Random Forest algorithm (background: 1, foreground: 2)
        mask_coarse += 1
        mask_refined = randomForestSegmentation(img, mask_coarse)

        # Plot results
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
        ax[0].imshow(segmentation.mark_boundaries(img, mask_refined, mode='thick'))
        ax[0].contour(mask_coarse)
        ax[0].set_title('Image, mask and segmentation boundaries')
        ax[1].imshow(mask_refined)
        ax[1].set_title('Segmentation')
        fig.tight_layout()
        plt.show()
    
        # Saving the binary mask
        output_path = args.output_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.imsave(os.path.join(output_path, f'{im}_binary.jpg'), mask_refined, cmap='binary')

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='GrubCut with RandomForest segmentation algorithm')
    parser.add_argument('-i', '--input_dir', default='./images',type=str,
                        help='Path to the input images folder (default: ./images)')
    parser.add_argument('-o', '--output_dir', default='./outputs',type=str,
                        help='Path to the input images folder (default: ./outputs)')
    args = parser.parse_args()

    main(args)
    