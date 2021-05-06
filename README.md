# STSIM

Implementation of Pytorch version of STSIM metrics.

Right now for STSIM-1 and STSIM-2 with steerable complex filter (SCF), I am actually using the magnitude of SCF,
because my Pytorch version 1.6.0 doesn't support complex number. Version >= 1.8.0 supports, but that doesn't support my GTX3080.

When apply steerble filter on STSIM-1 and STSIM-2 it works, because steerble filter gives real coefficients and Pytorch can handle that.

## To-do List:

Rewrite STSIM-I

Validate the choice of rank by SVD

Train and test with Corbis dataset, train with Corbis and test on Jana's dataset

## Reference Github Repo:
[steerable pyramid filters](https://github.com/LabForComputationalVision/pyPyrTools)

[DISTS](https://github.com/dingkeyan93/DISTS)

## Reference Paper:
[Structural Texture Similarity Metrics for Image Analysis and Retrieval](http://users.eecs.northwestern.edu/~pappas/papers/zujovic_tip13.pdf)

[Subjective and Objective Texture Similarity for Image Compression](https://www.researchgate.net/profile/Huib_Ridder/publication/261466382_Subjective_and_objective_texture_similarity_for_image_compression/links/54d38b270cf2b0c6146daf4b.pdf)

## Usage:

`python train.py`

`python test.py`

## Download Dataset:

TBD
