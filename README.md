# Multi Context based-Knowledge-distilation

Among these the knowledge distillation, is a general purpose technique that at first glance is widely applicable and complements all other ways of compressing neural networks . The key idea is to use soft probabilities (or ‘logits’) of a larger “teacher network” to supervise a smaller “student” network, in addition to the available class labels. These soft probabilities reveal more information than the class labels alone, and can purportedly help the student network learn better.

![Knowledge distilation Strucutre](https://github.com/glthrivikram/Multistudent-Knowledge-distilation/blob/main/images/Knowledge%20distilation%20structure.png)

# Proposed architecture
![proposed](https://github.com/glthrivikram/Multistudent-Knowledge-distilation/blob/main/images/MS.png)

 

# Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

![Dataset](https://github.com/glthrivikram/Multistudent-Knowledge-distilation/blob/main/images/cifar10.png)

# Teacher - Resnet50

ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks.The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks with 150+layers successfully. Prior to ResNet training very deep neural networks was difficult due to the problem of vanishing gradients.This problem was addressed using the concept of skip connections.

![resnet50-meta](https://github.com/glthrivikram/Multistudent-Knowledge-distilation/blob/main/images/resnetmeta.png)


# Training the Teacher

```bat
   python teacherTrain.py 
```
# Student Models

**student 1 - DenseNet121**
  
   DenseNet (Dense Convolutional Network) is an architecture that focuses on making the deep learning networks go even deeper, but at the same time making them more efficient to    train, by using shorter connections between the layers. DenseNet is a convolutional neural network where each layer is connected to all other layers that are deeper in the      network, that is, the first layer is connected to the 2nd, 3rd, 4th and so on, the second layer is connected to the 3rd, 4th, 5th and so on
![DenseNet](https://miro.medium.com/max/875/1*B0urIaJjCIjFIz1MZOP5YQ.png)


**student 2 - GoogleNet**
 

steps to train student using KD 

selector model - why , how para and model 

steps to train 

inferencing 


results 
