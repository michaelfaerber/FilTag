# FilTag
***
This directory contains files corresponding to the paper Explaining Convolutional Neural Networks by Tagging Filters.
It features FilTag which tags filters with class labels to support convolutional neural network explainability.

## code
***
The approach consists of four steps:
* [step1collectingactivations.py]: computes the activations of feature maps based on a data set and a cnn (examplary for ImageNet and VGG16, other cnns can also be executed).
* [step2labelfilters.py]: assigns the tags to the filters according to our approach, corresponding to the activations of the feature maps assigned to the filters.
* [step3evaluation.py]: computes predictions of the evaluation method.

## execution
***
For execution, the path to the images of the corresponding data set needs to be specified. 

## additional files
***
* [EXAMPLE_FilterTags_VGG16_conv5-3_k1.txt]: contains filter tags for the last convolutional layer of VGG16 with k=1.
* [decoding_wnids.txt]: contains an assignment from wnid to label for ImageNet
 to decode the labels.

