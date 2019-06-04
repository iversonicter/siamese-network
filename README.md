# SIAMESE NETWORK

## Introduction

This repo just illustrates the mechanism of contrastive loss function. 

Under this loss, if x_i, x_j are the features from same person, the distance is small. If x_i, x_j are the features from different person, the distance is large. 

![loss](https://github.com/iversonicter/siamese-network/blob/master/data/loss.png)

The network structure can be illustrated in the following image:

![network](https://github.com/iversonicter/siamese-network/blob/master/data/siamese.png)

Download the att face training set, and create your own positive pairs and negative pairs. The ratio should be 1:1 ideally, but in fact, the negative is far more than positive pairs. Hard sample mining is adopted to make learning efficient.

# Training result

positive:negative = 1:1
learning rate = 0.0005

![Training loss](https://github.com/iversonicter/siamese-network/blob/master/log/figure_0.0005_1_1.png)

