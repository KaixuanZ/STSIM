# STSIM

Implementation of Pytorch version of STSIM metrics.

## To-do List:

Understand which label to use and why

### Zhaochen

Estimating the parameters of STSIM-M & STSIM-I by using the variance of each feature in training set

Learning the parameters by data & label (Mahalanobis distance, LFDA, and Linear regression)

### Kaixuan

Comparing with other recent year's papers (deep learning based)

Comparing these methods on other datasets (if needed)


## Current Progress:
Steerable pyramid filters finished.

STSIM-M feature extraction finished.

STSIM-1 global, STSIM-2 global finished.

Test with Borda's rule.

## Reference Github Repo:
[steerable pyramid filters](https://github.com/LabForComputationalVision/pyPyrTools)

## Reference Paper:
[Structural Texture Similarity Metrics for Image Analysis and Retrieval](http://users.eecs.northwestern.edu/~pappas/papers/zujovic_tip13.pdf)

[Subjective and Objective Texture Similarity for Image Compression](https://www.researchgate.net/profile/Huib_Ridder/publication/261466382_Subjective_and_objective_texture_similarity_for_image_compression/links/54d38b270cf2b0c6146daf4b.pdf)

## Demo:
[main.py](https://github.com/KaixuanZ/STSIM/blob/master/main.py)
