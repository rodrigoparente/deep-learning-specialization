# LeNet-5, AlexNet, VGG-16

**LeNet-5**

- Developed by Yann LeCun (1998)
- Designed for handwritten digit recognition (e.g., MNIST dataset)
- It's architecture is composed of:
  - Input: 32x32 grayscale images
  - Convolutional Layers: Two convolutional layers followed by subsampling (average pooling)
  - Fully Connected Layers: Two fully connected layers
  - Output Layer: 10 classes (0-9 digits)
- Key Features:
  - Early use of convolutional layers combined with subsampling.
  - Designed for recognizing simple patterns, like digits, with minimal computational resources.

**AlexNet**

- Developed by Alex Krizhevsky et al. (2012)
- Designed for image classification on ImageNet (ILSVRC 2012)
- It's architecture is composed of:
  - Input: 224x224 RGB images
  - Convolutional Layers: Five convolutional layers, with max pooling after some layers
  - Fully Connected Layers: Three fully connected layers, including the output layer
  - Output Layer: 1000 classes (ImageNet)
- Key Features:
  - Introduced ReLU activation, dropout, and data augmentation.
  - First model to utilize GPUs for training, leading to a significant reduction in training time.
  - Achieved breakthrough performance on large-scale image classification tasks.

**VGG-16**

- Developed by Visual Graphics Group (VGG) at Oxford University (2014)
- Designed for image classification on ImageNet
- It's architecture is composed of:
  - Input: 224x224 RGB images
  - Convolutional Layers: 13 convolutional layers, using small 3x3 filters, followed by max pooling
  - Fully Connected Layers: Three fully connected layers
  - Output Layer: 1000 classes (ImageNet)
- Key Features:
  - Demonstrated the effectiveness of using deep networks with small filter sizes.
  - Known for its simplicity and depth, making it a popular choice for transfer learning.
  - Requires significant computational resources due to its large number of parameters.