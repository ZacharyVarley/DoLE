# DoLE

First public commit of difference of local entropy pre-processing filter for SIFT-style keypoint detection. Useful for
multimodal image registration. The idea for keypoint detection is to build an image scale space similar to the
difference-of-Gaussian (DoG). The smoothing operation that replaces a Gaussian kernel convolution is a local entropy
filter densely computed across the image in the spectral domain (2D real-valued FFTs) or using a separable window 
(a box instead of a unit disk) and image integrals.

An implementation of direct linear transform (DLT) for affine transform estimation is also provided. Many of the materials 
science use cases that I am applying this approach to do not require a homograph for a majority of the explain deformation.

Accompanying this keypoint code is a homography fine-tuning implementation of modality independent neighborhood descriptor
(MIND) evaluated using an image pyramid with a Lie algebra representation of the homograph. This representation is 
particularly useful for constraining homographs (e.g. no shearing), and for removing order-dependence of the group 
members. For rigid motion and similarity transformations, the matrix exponential has a closed form, but this is not the 
case for an affine transform or homograph.