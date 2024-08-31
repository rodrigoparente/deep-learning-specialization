# Transfer Learning

 - Transfer learning involves applying knowledge from one Task A to improve performance on a different but related Task B.
 - For instance, a neural network trained for image recognition can be adapted for radiology diagnosis.
 
**The Steps to Transfer Learning**

 - **Pre-training:** Train the neural network on Task A, such as image recognition.
 - **Transfer:** Replace the output layer(s) of the network with new layers for Task B, such as radiology diagnosis.
 - **Fine-tuning:** Retrain the network on Task B data. 
 
If Task B has limited data, you might only retrain the new layers; with more data, you can fine-tune the entire network.

**When Transfer Learning Makes Sense**

- Transfer learning proves beneficial when Task A has a large dataset, while Task B has limited data.
- It's effectiveness increases when both tasks use the same input type, such as images or audio.
- Is particularly useful if low-level features learned from Task A, like edge detection in images, are applicable to Task B.

# Multi-task Learning

 - Multi-task Learning involves training a single neural network to perform multiple tasks simultaneously.
 - The network learns to solve several problems at once, with each task potentially improving the performance on the others.
 - For example, in autonomous vehicles, a self-driving car needs to detect various objects such as pedestrians, other cars, stop signs, and traffic lights.
 - For an image with multiple objects, instead of a single label, the output could consists of multiple labels (e.g., presence or absence of pedestrians, cars, stop signs, and traffic lights).

**Training the Neural Network**

 - The network outputs a 4-dimensional prediction, $ \hat{y}(i) $, where each dimension corresponds to a specific object.
 - The loss function is computed by averaging the individual losses for each label across the entire training set:
    $$ \text{Loss} = \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{4} L(y_{ji}, \hat{y}_{ji}) $$
    where $ L $ represents the logistic loss function:
    $$ L(y_{ji}, \hat{y}_{ji}) = -y_{ji} \log(\hat{y}_{ji}) - (1 - y_{ji}) \log(1 - \hat{y}_{ji}) $$
 - Unlike softmax regression, where each image is assigned a single label, multi-task learning allows multiple labels per image.

**When Multi-task Learning Makes Sense**

 - It is beneficial when tasks share low-level features.
 - It is advantageous when the amount of data is similar across tasks.
 - A sufficiently large neural network is required.
