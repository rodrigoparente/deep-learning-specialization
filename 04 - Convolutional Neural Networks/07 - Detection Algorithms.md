# Object Localization

 - Standard image classification involves predicting the class label of an image using a neural network.
 - Object localization focuses on determining the location of an object within an image.
 - The neural network must predict the bounding box coordinates, which include:
    - $ b_x $: x-coordinate of the center of the bounding box
    - $ b_y $: y-coordinate of the center of the bounding box
    - $ b_h $: height of the bounding box
    - $ b_w $: width of the bounding box
 - These coordinates are normalized relative to the image dimensions, with (0,0) representing the top-left corner and (1,1) representing the bottom-right corner.

**Target Label $ y $ for Supervised Learning**

 - Consider a classification problem with class 1 for pedestrian, 2 for cars, 3 for motorcycles.
 - The target label vector has a dimension of $ (8, 1) $ and includes:
    - $ p_c $ indicates the presence of an object (1 if present, 0 if not).
    - $ b_x, b_y, b_h, b_w $ are bounding box parameters if $ p_c = 1 $.
    - $ c1, c2, c3 $ are class indicators for the detected object.
 - Label examples:
    - **Image with a Car**:
        $$
        y = \begin{bmatrix}
        1 \\
        b_x \\
        b_y \\
        b_h \\
        b_w \\
        0 \\
        1 \\
        0
        \end{bmatrix}
        $$
    - **Image with No Object**:
        $$
        y = \begin{bmatrix}
        0 \\
        ? \\
        ? \\
        ? \\
        ? \\
        ? \\
        ? \\
        ?
        \end{bmatrix}
        $$


**Loss Function for Training**

 - Squared Error Loss is commonly used to minimize the difference between predicted and true bounding box coordinates.
 - **When $ p_c = 1 $ (an object is present)**:
    - The loss is computed as the sum of squared differences between the predicted and true values for all relevant components:
      $$
      \text{Loss} = (b_x^\text{hat} - b_x)^2 + (b_y^\text{hat} - b_y)^2 + (b_h^\text{hat} - b_h)^2 + (b_w^\text{hat} - b_w)^2 + (c1^\text{hat} - c1)^2 + (c2^\text{hat} - c2)^2 + (c3^\text{hat} - c3)^2
      $$
    - This loss function ensures that the model not only classifies the object correctly but also accurately predicts its location within the image.
    - Minimizing this loss helps the model to improve both object localization and classification accuracy.

  - **When $ p_c = 0 $ (no object is present)**:
    - Only the error in predicting $ p_c $ is considered, as other components become irrelevant:
      $$
      \text{Loss} = (p_c^\text{hat} - p_c)^2
      $$
    - When no object is present, the model should focus solely on correctly predicting the absence of an object.
    - This loss function helps in fine-tuning the model to avoid false positives, which is crucial for accurate object detection.

# Landmark Detection

 - Neural networks can be adapted to output $X$ and $Y$ coordinates of significant points, or "landmarks," in an image.
 - A large labeled dataset with annotated landmarks is essential for training models for landmark detection.

**Applications of Landmark Detection**

 - Detecting emotions from facial expressions using facial landmarks.
 - Applying AR filters (e.g., Snapchat) to overlay effects like crowns or hats.
 - Identifying key points in a person’s pose, such as the chest, shoulders, elbows, and wrists.

**Landmark Detection Example**

 - In face recognition, the network can predict the coordinates of specific facial features, such as:
    - The coordinates for the corner of an eye.
    - Multiple key points across the eyes, mouth, nose, and jawline.
 - After processing the image through a convolutional neural network (ConvNet), the final layer outputs:
    - 1 unit to indicate the presence of a face (0 or 1).
    - $n$ units for $n/2$ landmarks, each providing $l_x$ and $l_y$ coordinates.

# Object Detection

 - Object detection can be achieved using a ConvNet with the Sliding Windows Detection Algorithm.
 - The goal is to identify all instances of a specific object class (e.g., cars) within an image by scanning different regions.

**Steps to Object Detection**
  
 - Apply the trained ConvNet to a test image by selecting a small rectangular region (window) within the image.
 - The ConvNet classifies whether this region contains the object of interest (e.g., a car).
 - Slide the window across the entire image, making predictions at each position.
 - Repeat the process with larger windows to capture objects of varying sizes.
 - Continue this process with different window sizes and positions to cover the entire image.

**Computational Cost**

 - The Sliding Windows method is computationally expensive because it requires running the ConvNet multiple times for different regions.
 - Using a coarse stride (larger step size) reduces the number of windows but may decrease detection accuracy.
 - Using a fine stride (smaller step size) increases computational cost but improves accuracy.

# Convolutional Implementation of Sliding Windows

 - Traditional sliding windows methods are inefficient because they require running the Convolutional Neural Network (ConvNet) separately on multiple overlapping regions of the image.
 - So, instead of applying the ConvNet multiple times on different image regions, you can use a convolutional approach to process the entire image in one go.
 - This method leverages the convolutional operation to simultaneously handle overlapping regions of the image.
 - By sharing computations across these regions, it reduces the need for redundant calculations.

**How It Works**

 - **Traditional Sliding Windows:**
    - The ConvNet is applied repeatedly to small regions of the image (e.g., $ R \times R $ areas) by sliding a window across the image.
    - Each windowed region is processed separately, resulting in multiple passes over overlapping areas of the image.
    - This approach is computationally expensive because it requires running the ConvNet multiple times, leading to redundant calculations in overlapping regions.

 - **Convolutional Implementation:**
    - Instead of processing each small region separately, the ConvNet is applied to the entire image in a single pass.
    - The ConvNet uses convolutional layers that naturally slide over the image, processing multiple regions simultaneously.
    - This results in an output feature map (or volume) where each point in the map corresponds to the output of the ConvNet for a specific region of the image.
    - For example, if the ConvNet has an input of $ H \times W \times C $ (height, width, channels) and uses a filter of size $ R \times R $, the output might be $ P \times P \times F $, where $ P $ is the spatial dimension of the feature map and $ F $ is the number of filters used.
    - Each $ 1 \times 1 \times F $ section of the output map represents the ConvNet's output for a particular $ R \times R $ region of the image.
    - By processing the entire image at once, overlapping computations are shared, significantly reducing the number of redundant calculations.

# Bounding Box Predictions

 - The YOLO (You Only Look Once) algorithm is a real-time object detection system that stands out for its speed and accuracy. 
 - It is a fully convolutional neural network (CNN) that divides the input image into a grid and predicts bounding boxes and probabilities for each grid cell.
 - YOLO treats object detection as a single regression problem, directly predicting bounding boxes and their corresponding class probabilities from full images in one evaluation.

**How It Works**

 - **Grid Division**:
   - The input image is divided into an $ S \times S $ grid.
   - Each grid cell is responsible for detecting objects whose center falls within the cell.
   - Each grid cell predicts a fixed number $ B $ of bounding boxes, confidence scores, and class probabilities.

 - **Bounding Box Prediction**:
   - Each bounding box is defined by at least 6 parameters:
     $$ \text{Bounding Box} = (p_c, b_x, b_y, b_h, b_w, c) $$
     Where:
     - $ p_c $ indicates the presence of an object (1 if present, 0 if not).
     - $ b_x, b_y, b_h, b_w $ are bounding box parameters if $ p_c = 1 $.
     - $ c $ is the class indicator for the detected object.

# Intersection Over Union

 - IoU is a function used to evaluate the accuracy of an object detection algorithm by comparing the predicted bounding box with the ground-truth bounding box.
 - It measures how well the predicted bounding box overlaps with the actual object.
 - To calculate the value of $ \text{IoU} $ you can use the following formula:
    $$  \text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}} $$
    where:
     - **Area of Intersection:** The area where the predicted bounding box overlaps with the ground-truth bounding box.
     - **Area of Union:** The total area covered by both the predicted and ground-truth bounding boxes.
 - By convention, an IoU value of $0.5$ or higher is considered a good match, indicating that the predicted bounding box is sufficiently accurate.

# Non-max Suppression (NMS)

 - Object detection algorithms may produce multiple bounding boxes for the same object, leading to redundant detections.
 - NMS ensures that each object is detected only once by suppressing redundant bounding boxes that represent the same object.

**How NMS Works**
  
  - **Step 1: Initial Predictions**:
    - Each grid cell outputs a probability $ p_c $ indicating the likelihood of an object and a bounding box.
    - Due to overlapping grid cells, multiple bounding boxes may overlap significantly, representing the same object.

  - **Step 2: Select the Highest Probability Box**:
    - Identify the bounding box with the highest probability $ p_c $.
    - This box is selected as the most confident prediction for that object.

  - **Step 3: Suppress Overlapping Boxes**:
    - Compute the Intersection over Union (IoU) between the selected bounding box and all other bounding boxes.
    - Suppress (discard) any boxes that have a high IoU with the selected box, as they likely represent the same object.

  - **Step 4: Repeat Process**:
    - Repeat the process by selecting the next highest probability box from the remaining (non-suppressed) boxes.
    - Continue suppressing overlapping boxes until no boxes remain or all have been processed.

# Anchor Boxes

 - Traditionally, if multiple objects fall within the same grid cell, the grid cell cannot handle multiple detections.
 - To address the limitation, the concept of anchor boxes is introduced. 
 - Anchor boxes are predefined shapes (e.g., rectangles of different aspect ratios) that allow multiple detections within the same grid cell.
 - Typically, two or more anchor boxes are used.

**How Anchor Boxes Works**

  - Instead of a single output vector $ \textbf{Y} $ per grid cell, we now have two vectors associated with two anchor boxes.
  - Each anchor box has its own set of outputs: $ p_c, p_x, p_y, p_h, p_w, C_1, C_2, C_3 $ (eight values per anchor box).
  - Thus, the output for each grid cell becomes:
    $$ 
        \textbf{Y} = \begin{bmatrix}
        p_{c1} \\
        p_{x1} \\
        p_{y1} \\
        p_{h1} \\
        p_{w1} \\
        C_{11} \\
        C_{21} \\
        C_{31} \\
        p_{c2} \\
        p_{x2} \\
        p_{y2} \\
        p_{h2} \\
        p_{w2} \\
        C_{12} \\
        C_{22} \\
        C_{32}
        \end{bmatrix}
    $$
 - For each object in the grid cell, calculate the Intersection over Union (IoU) with each anchor box.
 - We then assign the object to the anchor box with the highest IoU. 

# Semantic Segmentation with U-Net

 - Semantic Segmentation is a computer vision technique that classifies each pixel in an image into predefined categories.
 - Unlike object detection, which uses bounding boxes, semantic segmentation assigns a class label to every pixel, creating a detailed, pixel-level map of the image.
 - Its useful, for example, in:
    - Identifying drivable areas, road boundaries, and obstacles by labeling every pixel.
    - Segmenting anatomical structures (e.g., organs, tumors) in scans for diagnosis and surgical planning.

**How Semantic Segmentation Works**

  - **Standard CNN Architecture**
    - In traditional CNNs, the image dimensions decrease through layers and the final output is a class label.

  - **Semantic Segmentation**:
    - Instead of shrinking dimensions, the architecture needs to expand them. to reconstruct the full-size image.
    - The output is a full-size image with each pixel labeled according to its class.

# Transpose Convolution

 - A transpose convolution operation can be thought of as the reverse of a standard convolution operation.
 - While a standard convolution operation reduces the dimensions of an input, a transpose convolution increases it.
 - It is used in various neural network architectures for tasks such as image generation and segmentation.

**How Transpose Convolution Works**

 1. **Zero Padding:**
    - Insert zeroes between the elements of the input tensor to create a larger intermediate tensor.
    - This intermediate tensor will have more rows and columns than the original input, based on the stride and filter size.

 2. **Filter Application:**
    - Place the filter on different positions of the intermediate tensor to generate the output tensor.
    - Compute Values:
        - Multiply each value of the filter with corresponding values in the zero-padded intermediate tensor.
        - Place the result in the output tensor according to the stride and position of the filter.

 3. **Handle Overlaps:**
    - Where multiple filter applications overlap in the output tensor, sum the overlapping values.
    - This sum provides the final result for those overlapping positions.

 4. **Result:**
    - The result is a larger tensor with dimensions increased from the original input tensor.
    - The size of the output tensor depends on the filter size, stride, and padding used.
