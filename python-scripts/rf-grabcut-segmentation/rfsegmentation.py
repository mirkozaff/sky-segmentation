from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial

def randomForestSegmentation(img, training_labels):
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

    return result