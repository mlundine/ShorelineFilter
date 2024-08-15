# ShorelineFilter

# Image Suitability Filter

This is a deep learning model that sorts out unsuitable imagery for shoreline extraction (among other applications). It was trained on over a million coastal satellite images from around the world. It uses an Xception-esque architecture. We need this model because:

1. Satellite imagery is noisy. Clouds get in the way, haze gets in the way, sea-ice gets in the way, color-spaces get warped, and sometimes there are data gaps.
 
2. When the imagery is noisy, we can't pull out shorelines or other interesting features. 

3. Trying to detect and mask out clouds and sea-ice just puts in more data gaps and confuses segmentation models. At best we get a partial shoreline, at worst we get a messy contour that we have to throw out later on. The first step is to filter out unsuitable imagery for shoreline extraction. 

4. We don't want to manually sort images anymore. This is time-consuming and hurts our eyes.

# Image Segmentation Filter

This is a deep learning model that sorts out bad segmentation outputs. It was trained on just over 2,000 images from around the world. It's a pretty basic 2D convolutional neural network. We need this because:
1. We will never have a segmentation model that works perfectly on every image. We don't have labels from every coastal region in the world, nor do we have labels from every possible time period. Thus the segmentation models have to extrapolate A LOT. Sometimes we get garbage outputs from the segmentation model which results in wacky shoreline contours. 
2. Without this, we have to either manually filter out erroneous segmentation and shoreline outputs or we have to come up with a bunch of other filters to clean up the shoreline outputs.

# Shoreline Change Envelope (KDE filter)

This is just a method for computing a heat map of the extracted shoreline points, and then converting that heat map into a polygon feature that can be used as a spatial filter for extracted shorelines. Basically, we remove erroneous shoreline points by only keeping where most of the shorelines extracted from a long timeseries of satellite imagery fall. Shoreline points way outside of the average location get thrown out.




