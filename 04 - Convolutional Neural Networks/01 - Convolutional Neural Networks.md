# Computer Vision

 - Advances in computer vision are enabling applications that were previously impossible.
 - Some examples of these applications are:
    - **Image Classification:** Recognizing objects (e.g., determining if an image is of a cat).
    - **Object Detection:** Not only recognizing objects (e.g., cars) but also determining their positions within an image (e.g., for self-driving cars to avoid them).
    - **Neural Style Transfer:** Merging the content of one image with the style of another (e.g., repainting an image in the style of a Picasso).

**Challenges of handling large input**

 - Smaller images (e.g., 64x64 pixels) have manageable input dimensions (64x64x3 = 12,288 features).
 - Larger images (e.g., 1000x1000 pixels) lead to significantly larger input dimensions (1000x1000x3 = 3 million features).
 - Large input dimensions result in massive parameter sizes for neural networks:
    - For a fully connected network with 1,000 hidden units, the weight matrix ($W_1$) would be 1,000x3 million, totaling 3 billion parameters.
    - Training a neural network with billions parameters is computationally and memory-intensive, making it infeasible without massive data and resources.

# Edge Detection

 - The convolution operation is a fundamental component of Convolutional Neural Networks (CNNs).
 - It is used to detect various patterns or features in input data, such as identifying edges in an image.
 - Consider an input matrix $ \mathbf{X} $ with dimensions $ n \times n $, representing an image.
 - For instance, this could be a grayscale image where each element in the matrix corresponds to the intensity of a pixel.

**Filter (Kernel)**
  
 - A filter, also called a kernel, is a smaller matrix $ \mathbf{F} $ of size $ f \times f $, where $ f < n $.
 - Example of a generic $ 3 \times 3 $ filter matrix:
    $$
    \mathbf{F} = \begin{pmatrix}
    f_{11} & f_{12} & f_{13} \\
    f_{21} & f_{22} & f_{23} \\
    f_{31} & f_{32} & f_{33}
    \end{pmatrix}
    $$
 - The values $ f_{ij} $ in the filter are designed to detect specific patterns in the input matrix.

**Convolution Operation**

 - The convolution operation involves sliding the filter $ \mathbf{F} $ over the input matrix $ \mathbf{X} $ and performing an element-wise multiplication between the filter and the corresponding sub-matrix of the input, with dimensions $ f \times f $.
 - The results of these multiplications are then summed to produce a single value in the output matrix $ \mathbf{Y} $.
 - This operation is repeated across the entire input matrix $ \mathbf{X} $ and is mathematically denoted as:
    $$
    \mathbf{Y} = \mathbf{X} * \mathbf{F}
    $$
   - Here, $ \mathbf{Y} $ is the resulting output matrix of dimensions $ (n - f + 1, n - f + 1) $.
 - The resulting matrix $ \mathbf{Y} $ will highlight the presence of the pattern defined by the filter $ \mathbf{F} $ within the input matrix $ \mathbf{X} $.
 - For example, if $ \mathbf{F} $ is designed to detect vertical edges, then $ \mathbf{Y} $ will have higher values where vertical edges are present in $ \mathbf{X} $.

# Filters

Here are examples of vertical edge detection filters commonly used in image processing:

 - **Basic Vertical Edge Detection Filter:**
     $$
     \mathbf{F} = \begin{pmatrix}
     1 & 0 & -1 \\
     1 & 0 & -1 \\
     1 & 0 & -1
     \end{pmatrix}
     $$
     - This filter detects vertical edges by emphasizing the difference between the left and right columns of pixels.
     - Positive values on the left side and negative values on the right highlight vertical transitions.

 - **Sobel Vertical Edge Detection Filter:**
     $$
     \mathbf{F}_{\text{sobel}} = \begin{pmatrix}
     1 & 0 & -1 \\
     2 & 0 & -2 \\
     1 & 0 & -1
     \end{pmatrix}
     $$
     - The Sobel filter emphasizes the central row more, making it more robust to noise.
     - It provides a stronger response to vertical edges.

 - **Scharr Vertical Edge Detection Filter:**
     $$
     \mathbf{F}_{\text{scharr}} = \begin{pmatrix}
     3 & 0 & -3 \\
     10 & 0 & -10 \\
     3 & 0 & -3
     \end{pmatrix}
     $$
     - The Scharr filter is another variation that gives even more weight to the central pixels.
     - It can improve edge detection in some scenarios.

To detect horizontal lines, you can rotate any of these filters vertically by $90^\circ$.

**Learning Filters with Neural Networks**

 - Rather than manually designing filters, deep learning allows these filters to be learned from data through backpropagation.
 - The neural network can learn optimal filters for edge detection at various angles (not just vertical or horizontal) and for more complex features.
 - This approach can lead to more accurate and robust feature detection compared to hand-coded filters.
 - In this process, the $3 \times 3$ filter consists of 9 parameters that the network learns and adjusts during training to effectively capture the relevant features in the data.

# Padding

 - Each convolution operation reduces the image dimensions, potentially shrinking it to a very small size after multiple layers.
 - Pixels near the edges or corners are used less frequently in the output, leading to loss of edge information.
 - Padding involves adding extra pixels (typically zeros) around the border of the image before applying the convolution.
 - The dimensions of the output matrix $ \mathbf{Y} $ are given by:
    $$ \mathbf{Y}_{(h, w)} = \bigg( n - f + 2p + 1, n - f + 2p + 1 \bigg) $$
    Here:
    - $ n $ = size of the input matrix $ \mathbf{X} $
    - $ f $ = size of the filter matrix $ \mathbf{F} $
    - $ p $ = padding value
 - Filters are typically odd-sized to ensure symmetric padding around the central pixel.

# Strided Convolutions

 - Stride convolutions involve moving the filter across the input image with steps larger than one.
 - This changes the spacing between positions where the filter is applied.
 - The size of the output matrix $ \mathbf{Y} $ is given by:
    $$ \mathbf{Y}_{(h, w)} = \bigg( \bigg\lfloor \frac{n - f + 2p}{s} \bigg\rfloor + 1, \bigg\lfloor \frac{n - f + 2p}{s} \bigg\rfloor + 1 \bigg)  $$
    where:
    - $ n $ = size of the input $ \mathbf{X} $
    - $ f $ = size of the filter $ \mathbf{F} $
    - $ p $ = padding
    - $ s $ = stride

# Convolutions over 3D Volumes

 - Convolution operations can be applied to 3D volumes, extending beyond 2D images.
 - This approach is useful for analyzing multi-channel data such as color images or volumetric data.
 - For example, consider an image with dimensions $ m \times m \times d $, where $ d $ represents the number of channels (e.g., RGB channels).
 - You can use a 3D filter of size $ n \times n \times d $ for the convolution operation.
 - The size of the output matrix $ \mathbf{Y} $ is given by:
    $$ \mathbf{Y}_{(h, w, k)} = \bigg( \bigg\lfloor \frac{n - f + 2p}{s} \bigg\rfloor + 1, \bigg\lfloor \frac{n - f + 2p}{s} \bigg\rfloor + 1, \bigg\lfloor \frac{n - f + 2p}{s} \bigg\rfloor + 1 \bigg)  $$
    where:
    - $ n $ = height and width of the input volume
    - $ f $ = height and width of the filter
    - $ p $ = padding
    - $ s $ = stride

**Using Multiple Filters**

  - To detect various features (e.g., edges, textures), use multiple filters.
  - Each filter is designed to detect different features.
  - The outputs from all filters are stacked to create a 3D output volume.


# One Layer of a Convolutional Network

 - The steps to add a convolutional layer to a neural network are:
    1. **Convolve Input with Filters**: Apply filters to a 3D input, producing 2D output matrices.
    2. **Add Bias**: Add a scalar bias to each element of the output matrices.
    3. **Apply Non-Linearity**: Apply an activation function (e.g., ReLU) to the biased outputs.
    4. **Stack Outputs**: Combine the output matrices into a 3D volume.
 - This forms one convolutional layer in a CNN.

**Convolution Layer Notation**

 - $ f^{[l]} $: Filter size for layer $ l $ (e.g., 3x3).
 - $ p^{[l]} $: Padding applied in layer $ l $ (can be "valid" for no padding or "same" for equal output size).
 - $ s^{[l]} $: Stride for layer $ l $.
 - Input dimensions: $ n^{[l-1]}_h \times n^{[l-1]}_w \times n^{[l-1]}_c $, where $ h $, $ w $, and $ c $ denote height, width, and number of channels.
 - Output dimensions: $ n^{[l]}_h \times n^{[l]}_w \times n^{[l]}_c $.
 - The height and width of the output volume are calculated using the formula:
    $$ n^{[l]}_h = \left\lfloor \frac{n^{[l-1]}_h + 2p^{[l]} - f^{[l]}}{s^{[l]}} \right\rfloor + 1 $$
    The same formula applies for $ n^{[l]}_w $.

# Simple Convolutional Network Example

  - Here is an example of a Convulational Network for image classification (e.g., "Is this a cat?").
  - Consider the input image size is $ 39 \times 39 \times 3 $ (Height $ n_H $, Width $ n_W $, Channels $ n_C $).

**Layer 1: First Convolutional Layer**

- **Filters and Hyperparameters**:
  - Filter size $ f_1 = 3 \times 3 $
  - Stride $ s = 1 $
  - Padding $ p = 0 $
  - Number of filters $ n_C^{[1]} = 10 $

- **Output Dimension Calculation**:
  - Output height and width:
    $$ n_H^{[1]} = n_W^{[1]} = \left\lfloor \frac{n_H + 2p - f_1}{s} \right\rfloor + 1 = \left\lfloor \frac{39 + 0 - 3}{1} \right\rfloor + 1 = 37 $$
  - Output volume: $ 37 \times 37 \times 10 $

**Layer 2: Second Convolutional Layer**

- **Filters and Hyperparameters**:
  - Filter size $ f_2 = 5 \times 5 $
  - Stride $ s = 2 $
  - Padding $ p = 0 $
  - Number of filters $ n_C^{[2]} = 20 $

- **Output Dimension Calculation**:
  - Output height and width:
    $$ n_H^{[2]} = n_W^{[2]} = \left\lfloor \frac{n_H^{[1]} + 2p - f_2}{s} \right\rfloor + 1 = \left\lfloor \frac{37 + 0 - 5}{2} \right\rfloor + 1 = 17 $$
  - Output volume: $ 17 \times 17 \times 20 $

**Layer 3: Third Convolutional Layer**

- **Filters and Hyperparameters**:
  - Filter size $ f_3 = 5 \times 5 $
  - Stride $ s = 2 $
  - Padding $ p = 0 $
  - Number of filters $ n_C^{[3]} = 40 $

- **Output Dimension Calculation**:
  - Output height and width:
    $$ n_H^{[3]} = n_W^{[3]} = \left\lfloor \frac{n_H^{[2]} + 2p - f_3}{s} \right\rfloor + 1 = \left\lfloor \frac{17 + 0 - 5}{2} \right\rfloor + 1 = 7 $$
  - Output volume: $ 7 \times 7 \times 40 $

**Layer 4: Flatten Layer**

 - Flatten the $ 7 \times 7 \times 40 $ output into a vector of size $ 7 \times 7 \times 40 = 1960 $.
  
**Layer 5: Fully Connected Layer**
 
 - Use the flattened vector as input to a logistic regression or softmax layer for classification.

# Pooling Layers

 - Max pooling is a down-sampling technique used in Convolutional Neural Networks (CNNs) to reduce the spatial dimensions (height and width) of input feature maps while preserving important information.
 - It operates by sliding a window over the input feature map and selecting the maximum value within each window.
 - The selected maximum values are then placed in the corresponding positions of the output feature map.
 - This technique decreases the computational load by reducing the number of parameters in the network.
 - Additionally, max pooling enhances the model's ability to detect features regardless of small translations or distortions in the input.

**Example**

 - For a $ 2 \times 2 $ pooling window and a stride of 2:
 - If the input feature map $ X $ is:
  $$
  \begin{bmatrix}
  1 & 3 & 2 & 4 \\
  5 & 6 & 7 & 8 \\
  9 & 10 & 11 & 12 \\
  13 & 14 & 15 & 16
  \end{bmatrix}
  $$
 - The pooling operation will involve taking maximum values from each $ 2 \times 2 $ window:

  $$
  \text{Max}\left(\begin{matrix}
  1 & 3 \\
  5 & 6
  \end{matrix}\right) = 6
  $$

  $$
  \text{Max}\left(\begin{matrix}
  2 & 4 \\
  7 & 8
  \end{matrix}\right) = 8
  $$

  $$
  \text{Max}\left(\begin{matrix}
  9 & 10 \\
  13 & 14
  \end{matrix}\right) = 14
  $$

  $$
  \text{Max}\left(\begin{matrix}
  11 & 12 \\
  15 & 16
  \end{matrix}\right) = 16
  $$
 - The resulting output feature map $ Y $ is:
  $$
  \begin{bmatrix}
  6 & 8 \\
  14 & 16
  \end{bmatrix}
  $$

# Why Convolutions?

Convolutions are widely used in neural networks for several compelling reasons:

1. **Parameter Sharing:**

   - Convolutions apply the same filters across the entire image, significantly reducing the number of parameters.
   - This allows the network to detect features like edges or textures anywhere in the image, improving generalization.

2. **Sparsity of Connections:**

   - Each convolutional output is connected to a small local region, reducing the number of connections.
   - This focus on local features minimizes parameters and lowers the risk of overfitting.

3. **Translation Invariance:**

   - Convolutions make networks invariant to input shifts, recognizing features regardless of their position.
   - This enhances the network's robustness to variations in the input.
