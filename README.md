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

### Feeding the model
The next step is deciding how to feed the model. In this project, we decided to use the [tf.data API](https://www.tensorflow.org/api_docs/python/tf/data). This API allows us to create a Dataset from a TFRecord file. Once we have the image and labels as tensors, we distort the images to increase the training dataset and we normalize the images. You can find this operations in the method `distorted_input` of file `input.py`.

By default we are using batches of 32 elements, because it gave a good result when we trained the model. However you can change this parameter modifying the flag `batch_size`. Once the batches are created, we pass them to the model one batch each step. The data API can support different iterators, but we decided to use the [one shot Iterator](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#make_one_shot_iterator) in order to go over all the elements of the dataset.

## Training our model
The first step of the training is getting the bottlenecks of the previous old network. The bottlenecks are the values that are passed to the classifier, where all the image features have been computed. To do fine tuning, we must train a new classifier with the new classes. We build our classifier using a fully connected layer with 5 classes, which correspond to the 5 type of flowers that we have in our dataset.

For this project, we decided to use ADAM algorithm as the optimizer with an initial learning rate of 0,005. This hyperparameter can we easily changed on the FLAGS. ADAM minimizes progressively the error of the goal function, by changing the new classifier neuron's weights. The minimized loss is the mean of the losses in one bach and the loss of a single image is calculated using the function [`sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits)

### Visualizing the training
During the training, you can check how is going with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) that has been implemented in the project.

## Evaluation & Results
For evaluation we have used the eval dataset, and we have count how many images our neural network classify properly. We have used top-1 accuracy

Next we are going to show the results that we have obtained according to the steps and the hyperparameters. The execution time depends on whether it runs on CPU or GPU. Below we will explain how to execute the project in [Google Cloud](cloud.google.com/).

| Optimizer| Steps | Hyperparametres | Top-1 accuracy |
| ------------- | ------------- | ------------- |  ------------- |
Gradient Descent | 1500 | Exponential Learning rate starting at 0.4 | 79.83% |
Gradient Descent | 2000 | Exponential Learning rate starting at 0.4 | 83.08% |
Gradient Descent | 1500 | Exponential Learning rate starting at 0.2 | 82.56% |
ADAM |**500** | Default | **83.27 %** |
ADAM |1000 | Default | 82.22%  |
ADAM |500 | Learning rate = 0.005 | 82.19% |

## Running the model on Google Cloud
For those who do not have a GPU or do not want to have their computer busy training the network, there is the alternative to train it in the cloud. We recommend using the Machine Learning API of [Google Cloud](cloud.google.com/). 

First of all, you need to add two files to the project, the `config.yaml` and `setup.py`. The next step is to upload to Google Storage the data that our project needs, that is, the tfrecord file and the weights of our net. Make sure to have gcloud installed in your envieronment. Then, just run the following command on your project directory and the programm will start running on the cloud:

```
now=$(date +"%Y%m%d_%H%M%S")
export JOB_NAME="ft_flowers_$now"

gcloud ml-engine jobs submit training $JOB_NAME \
        --package-path main \
        --staging-bucket gs://{Your_Directory_Name} \
        --module-name main.train \
        --job-dir gs://{Your_Directory_Name}/$JOB_NAME/train_dir \
	      --region us-central1 \
        --config config.yaml \
        -- \
        --data_path gs://{Your_Directory_Name}/{Your_Folder_Name}/ \
        --ckpt_dir gs://{Your_Directory_Name}/{Your_Folder_Name}/checkpoints \
        --log_dir gs://{Your_Directory_Name}/$JOB_NAME/train_dir/logs \
        --save_dir gs://{Your_Directory_Name}/$JOB_NAME/train_dir/flower 
 ```

## Contribute!
If you find any bug in the repo, or if you think out a solution that could work better, feel free to open an issue in the project. The aim of this repo is to make easier to people learn how to do a fine tuning properly so we will be glad if more people contribute!

## Contact 
If you need to contact us in a faster way, you can do it via email:

* **Juan Abascal** - *juan.abascal@bq.com* 

* **Daniel Gonzalez** - *daniel.gonzalez@bq.com* 

