# Flower classificator
This is a Flower classificator built on the top of pretrained network for ImageNet. 
 In this example, you will learn how to fine tune a pretrained network with your own
categories. Unlike the example that you can find at [Tensorflow - Image retraining](https://www.tensorflow.org/tutorials/image_retraining) , 
this tutorial explains step by step how to make a fine tune correctly.

## Getting Started

First of all, you need to make sure you have the following things on your computer. We recommend using a virtual environment as a conda for installation

### Prerequisites
* Tensorflow >= 1.5
* Tensorboard (Optional but recommended)
* PIL

### Net & Weights
For this tutorial, we are using the Inception V3 architecture. You need to download the pretrained [Weights](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz).
You can use another network of those that appear in the following [link](https://github.com/tensorflow/models/tree/master/research/slim) 

### Dataset

We have decided to use the flowers dataset since they are creative-commons licensed flower photos and it is very easy to find many images for each class.
You can donwload the dataset [Here](http://download.tensorflow.org/example_images/flower_photos.tgz)
