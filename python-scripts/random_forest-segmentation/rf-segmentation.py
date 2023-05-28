import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial


def main(args):
    # Read input image
    img = plt.imread(args.input_image)

    # Build an array of labels for training the segmentation.
    training_labels = np.zeros(img.shape[:2], dtype=np.uint8)
    training_labels[180:, :] = 2
    training_labels[0:90, 0:100] = 1
    training_labels[0:80, 410:650] = 1

    sigma_min = 1
    sigma_max = 16
    # Feature extraction
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1)
    features = features_func(img)

    # RF training
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels, features, clf)

    # Calculate predictions
    result = future.predict_segmenter(features, clf)

    # Plot results
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
    ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
    ax[0].contour(training_labels)
    ax[0].set_title('Image, mask and segmentation boundaries')
    ax[1].imshow(result)
    ax[1].set_title('Segmentation')
    fig.tight_layout()
    plt.show()

    # Saving the binary mask
    plt.imsave(f'{args.input_image}_binary.jpg', result, cmap='binary')

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='GrubCut algorithm')
    parser.add_argument('-i', '--input_image', default='Immagine10.jpg',type=str,
                        help='Path to the input image (default: Immagine10.jpg)')
    args = parser.parse_args()

    main(args)
    