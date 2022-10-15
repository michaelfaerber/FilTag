# FilTag
This repository contains the code for the paper [Explaining Convolutional Neural Networks by Tagging Filters](https://arxiv.org/pdf/2109.09389.pdf) published at AIMLAI@CIKM'22.

## Abstract
```
Convolutional neural networks (CNNs) have achieved astonishing performance on various image classification tasks, but it is difficult for humans to understand how a classification comes about. Recent literature proposes methods to explain the classification process to humans. These focus mostly on visualizing feature maps and filter weights, which are not very intuitive for non-experts in analyzing a CNN classification. In this paper, we propose \textsc{FilTag}, an approach to effectively explain CNNs even to non-experts. The idea is that when images of a class frequently activate a convolutional filter, then that filter is tagged with that class. These tags provide an explanation to a reference of a class-specific feature detected by the filter. Based on the tagging, individual image classifications can then be intuitively explained in terms of the tags of the filters that the input image activates. Finally, we show that the tags are helpful in analyzing classification errors caused by noisy input images and that the tags can be further processed by machines.
```

## Code
The approach consists of four steps:
* [step1collectingactivations.py]: computes the activations of feature maps based on a data set and a cnn (examplary for ImageNet and VGG16, other cnns can also be executed).
* [step2labelfilters.py]: assigns the tags to the filters according to our approach, corresponding to the activations of the feature maps assigned to the filters.
* [step3evaluation.py]: computes predictions of the evaluation method.

## Execution
For execution, the path to the images of the corresponding data set needs to be specified. 

## Additional files
* [EXAMPLE_FilterTags_VGG16_conv5-3_k1.txt]: contains filter tags for the last convolutional layer of VGG16 with k=1.
* [decoding_wnids.txt]: contains an assignment from wnid to label for ImageNet to decode the labels.

## Contact & More Information
More information can be found in the paper [Explaining Convolutional Neural Networks by Tagging Filters](https://arxiv.org/pdf/2109.09389.pdf) (AIMLAI@CIKM'22).

## How to Cite
Please cite the work as follows:
```
@inproceedings{FilTag2022,
  author    = {Anna Nguyen and
               Daniel Hagenmayer and
               Tobias Weller and
               Michael F{\"{a}}rber},
  title     = {Explaining Convolutional Neural Networks by Tagging Filters},
  booktitle   = {Proc. of AIMLAI@CIKM'22},
  year      = {2022}
}
```