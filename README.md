<h1 style="text-align: center;">Deep Learning Fundamentals from Scratch</h1>

## What is this repo about?
This is a code base for my Deep Learning course in Fall 2022, an adapted version of the assignments for the ```CS231n: Deep Learning for Computer Vision``` from Stanford University, designed by Andrej Karpathy and Fei-Fei Li.<br>
We work through all the basic fundamentals and first principles of Deep Learning just using **Python and Numpy**.

This course consists of three assignment modules with each module going over a bunch of Deep Learning topics in depth.

## Assignment Modules
### Assignment 1
In this assignment you will practice putting together a simple image classification pipeline based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows

* Understand the basic Image Classification pipeline and the data-driven approach (train/predict stages).
* Understand the train/val/test splits and the use of validation data for hyperparameter tuning.
* Develop proficiency in writing efficient vectorized code with numpy.

This assignment module consists of the following topics<br>
1. Implement and apply a [k-Nearest Neighbor (kNN) classifier](/assignment1/knn.ipynb).
2. Implement and apply a [Multiclass Support Vector Machine (SVM) classifier](/assignment1/svm.ipynb).
3. Implement and apply a [Softmax](/assignment1/softmax.ipynb).
4. Implement and apply a [Two layer neural network](/assignment1/two_layer_net.ipynb). Understand the differences and tradeoffs between these classifiers.
5. Get a basic understanding of performance improvements from using [higher-level representations](/assignment1/features.ipynb) as opposed to raw pixels, e.g. color histograms, Histogram of Gradient (HOG) features, etc.

### Assignment 2
In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

* Understand Neural Networks and how they are arranged in layered architectures.
* Understand and be able to implement (vectorized) backpropagation.
* Implement various update rules used to optimize Neural Networks.

This assignment module consists of the following topics<br>
1. Implement and apply a [Multi-Layer Fully Connected Neural Networks](/assignment2/FullyConnectedNets.ipynb).
2. Implement [Batch Normalization and Layer Normalization](/assignment2/BatchNormalization.ipynb) for training deep networks.
3. Understand the architecture of [Convolutional Neural Networks](/assignment2/ConvolutionalNetworks.ipynb) and get practice with training them.
4. Implement [Dropout](/assignment2/Dropout.ipynb) to regularize networks.
5. Gain experience with a major deep learning framework, such as [TensorFlow](/assignment2/TensorFlow.ipynb) or [PyTorch](/assignment2/PyTorch.ipynb).

### Assignment 3
Here we learn more complex Deep Learning concepts. In this assignment, you will implement language networks and apply them to image captioning on the COCO dataset. You will explore methods for visualizing the features of a pretrained model on ImageNet and train a Generative Adversarial Network (GAN) to generate images that look like a training dataset. You will be introduced to self-supervised learning to automatically learn the visual representations of an unlabeled dataset.

This assignment module consists of the following topics<br>
1. Implement and apply [Image Captioning with Vanilla RNNs](/assignment3/RNN_Captioning.ipynb).
2. Implement and apply [Image Captioning with Transformers](/assignment3/Transformer_Captioning.ipynb).
3. Implement and apply [Network Visualization](/assignment3/Network_Visualization.ipynb): Saliency Maps, Class Visualization, and Fooling Images.
4. Implement and apply [Generative Adversarial Networks](/assignment3/Generative_Adversarial_Networks.ipynb).
5. Implement and apply [Self-Supervised Learning for Image Classification](/assignment3/Self_Supervised_Learning.ipynb)

<br>
Note: I have yet to push a couple of the assignments from this module.

For assignment3, I had to remove the datasets because they were too huge. You can download them all from [here](https://drive.google.com/drive/folders/1xpKckKMCBg7tjIc_HY5vQzK18_ZAL0mv?usp=drive_link) and place it in the following folder.
```
/assignment3/cs231n/datasets/
```
<br>


## Take-aways
I learned a lot about the basic building blocks and 'behind-the-scenes' for a lot of Deep Learning concepts that are often brushed aside as 'self explanatory'. This helped me when using frameworks like PyTorch and TensorFlow to understand what each of the functionalities actually do and how the affect the algorithm rather than just blindly using them in the code.

<!-- ## Usage
* Clone the repo to your local machine
```
git clone https://github.com/HemanthJoseph/Image-Stitching.git
```
* Change Directory
```
cd src
```
* Run the python file
```
python Image_Stitching.py
```

## Dependencies and libraries
1. Python 3.9.12
2. OpenCV '4.7.0'
3. Numpy '1.24.2' -->