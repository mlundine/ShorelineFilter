# ShorelineFilter

# Image Suitability Filter

This is a binary classification model that sorts out unsuitable imagery for shoreline extraction (among other applications). It was trained on over a million coastal satellite images from around the world. It uses an Xception-esque architecture.

4. We don't want to manually sort images anymore. This is time-consuming and hurts our eyes.

![training data](Figures/ImageSuitability/spatial_domain.png)

Above shows ROI locations where images were pulled from to train this model.

![optimum threshold](Figures/ImageSuitability/optimum_threshold.png)

Above shows some curves that tell us what threshold to run the models (RGB or grayscale) at to balance true and false positives. 

![RGB_model_obvious](Figures/ImageSuitability/RGB_model_obvious.jpg)

Above shows some example outputs from the test dataset where the model was extremely confident they were 'good' or 'bad'.

![rgb_model_edge](Figures/ImageSuitability/RGB_model_edge.jpg)

Above shows some example outputs from the test dataset where the model was not so confident (edge cases).

![test_scores_plot](Figures/ImageSuitability/test_scores_plot.png)

Above shows the distribution of sigmoid scores (probability of image being 'good') from the test dataset. It's bimodal and symmetric indicating the model learned how to distinguish the two classes very well.

![ROC_curve](Figures/ImageSuitability/ROC_curve.png)

Above shows the ROC curve. The RGB model is slightly more effective than the grayscale model.

# Image Segmentation Filter

This is a binary classification model that sorts out bad segmentation outputs. It was trained on just over 2,000 images from around the world. It's a pretty basic 2D convolutional neural network. We need this because:

# Shoreline Change Envelope (KDE filter)

This is just a method for computing a heat map of the extracted shoreline points, and then converting that heat map into a polygon feature that can be used as a spatial filter for extracted shorelines. Basically, we remove erroneous shoreline points by only keeping where most of the shorelines extracted from a long timeseries of satellite imagery fall. Shoreline points way outside of the average location get thrown out.

![kde filter](Figures/kde_filter.png)

Above shows the progression of this filter. We take our unfiltered extracted shoreline points for a particular ROI. Then we compute a spatial kernel density estimate of the points. We then convert this to a raster and then use Otsu thresholding to classify the raster into two categories (high density and low density). We then isolate the high density pixels and polygonise these pixels to get an envelope of shoreline change that we can use to filter erroneous points.

![shoreline_example](Figures/shoreline_example.png)

Above shows an example of using these filters on an ROI from western Alaska. A final filter involves taking the timeseries data and removing outliers and high-frequency noise with a median filter. We then re-project the cross-shore positions to geographic coordinates to get cleaner looking shorelines.




