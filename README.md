# Flower classificator
This is a Flower classificator built on the top of pretrained network for ImageNet. In this example, you will learn how to fine tune a pretrained network with your own categories. This example is based on [Tensorflow - Image retraining tutorial](https://www.tensorflow.org/tutorials/image_retraining), but here you will find an easier to follow code and we are using Tensorflow's new features like [tf.data API](https://www.tensorflow.org/api_docs/python/tf/data).

## Getting Started
First of all, you need to make sure you have the following things installed or downloaded on your computer. We also recommend using a virtual environment like [conda](https://www.anaconda.com/) or Python native [virtualenv](https://virtualenv.pypa.io/en/stable/) for creating isolated working enviroments.

### Prerequisites
* Python 3.6
* Tensorflow >= 1.5
* Tensorboard (Optional but recommended)
* PIL
* Numpy

### Downloads
You need to have the both the dataset and the net and weights downloaded in your computer:
* [Inception V3 weights](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
* [Flower dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)

## Net & Weights
In this example we are using the Google's Inception V3 architecture. The model is pretrained for ImageNet competition, which is able to classify images among 1001 classes with 93.9% top-5 accuracy.

Although we are using Inception V3 architecture, you can use whichever you want. You can find other kind of ImageNet pretrained architectures in this [tensorflow repo](https://github.com/tensorflow/models/tree/master/research/slim). **Be aware** that if you change the architecture you may change the input size of the images.

## Dataset
The dataset used for training consists in images of 5 different classes, placed in different folders from where the classes' names are taken:

```bash
├── flower_photos
│   ├── daisy
│   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips
```

The `pre_input.py` file helps you to create the TFRecord files that are used to feed the model in both training and eval mode. You have to run the script separately before executing the training or evaluation programme. The only files you need for doing the training and evaluation are the `training_set.tfrecord`, `eval_set.tfrecord` and `labels.txt`, so you could delete the images folder if you will.

### Creating training and evaluation sets
There are a total of 3.670 images, which are splitted into two datasets. Training dataset and eval dataset. You can adjust the number of images for each dataset by changing the `number_of_images_for_training` variable which is a global variable. This number **must be lower** than the minimun number of photos in a class, by default this number is 600. We recommend to use at least 25% of the images for evaluation purposes.

From the folder structure we create two different txt files, called `training_set.txt` and `eval_set.txt`. This files have in each line the path to each image in the dataset and its label, which is a number between 0 and 4. To generate this files you just have to run the `pre_input.py` file.

### Generating TFRecord files
We need to have a TFRecord file as input of our input pipeline. We generate this file from the training and eval dataset, resulting in two new files `training_set.tfrecord` and `eval_set.tfrecord`. You can learn more about the TFRecord format in [this Tensorflow's programmer's guide](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data).

In the TFRecord files we save the raw image’s pixels, its height and width and the label of the image. The sum of the size of `training_set.tfrecord` file and the `eval_set.tfrecord` might be larger than the size of all the images, because we are saving more information and also the images are not compressed as jpeg.
