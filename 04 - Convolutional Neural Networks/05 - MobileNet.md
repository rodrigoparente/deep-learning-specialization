# Depthwise Separable Convolution

**Depthwise Convolution**

 - Depthwise convolution reduces computational costs by convolving each input channel separately with its own filter.
 - For an input with $ D $ channels and $ k \times k $ filters, $ D $ individual convolutions are performed—one per channel.
 - Channels remain separate, unlike in standard convolution, and the output has the same number of channels as the input, with reduced spatial dimensions.

**Pointwise Convolution**

 - Pointwise convolution, a 1x1 convolution, follows depthwise convolution to combine features across channels.
 - It applies a 1x1 filter across all input channels, mixing their information to create new features.
 - The number of output channels is determined by the number of filters in the pointwise convolution.

**Depthwise + Pointwise Convolution**

 - Depthwise separable convolution combines depthwise and pointwise convolutions to reduce computation.
    1. **Depthwise Convolution**: Applies separate convolutions to each channel.
    2. **Pointwise Convolution**: Efficiently combines information across channels, replacing a full $ k \times k \times D $ convolution.
 - This techniques reduces computational cost and memory usage, while maintaining similar performance to standard convolutions.

# MobileNet V1

 - MobileNet V1 is a lightweight convolutional neural network designed for mobile and embedded vision applications.
 - Its key features include:
    - Replaces standard convolutions with depthwise separable convolutions to reduce computational cost and parameters.
    - Its architecture consists of a series of depthwise separable convolutions, followed by a few fully connected layers.
    - Achieves a good trade-off between model size and accuracy, making it suitable for resource-constrained environments.

# MobileNet V2

 - MobileNet V2 builds on the V1 architecture with several improvements:
    - Replaces the original non-linear activation functions with linear bottlenecks, improving performance and reducing information loss in the network.
    - Introduces inverted residual blocks that follow depthwise separable convolutions with a linear layer, enhancing feature reuse and reducing computational complexity.

# EfficientNet

 - EfficientNet is a family of convolutional neural networks optimized for efficiency and accuracy.
 - It employs a compound scaling method to uniformly adjust the network's depth, width, and resolution, balancing model size, computational cost, and accuracy.
 - The architecture begins with EfficientNet-B0, a compact and efficient baseline network.
 - This baseline is scaled up to create larger versions (e.g., B1, B2) with improved performance.
 - EfficientNet uses MobileNetV2’s MBConv blocks, which leverage depthwise separable convolutions and inverted residuals for enhanced efficiency.
 - The result is state-of-the-art accuracy with fewer parameters and lower computational costs compared to other networks.