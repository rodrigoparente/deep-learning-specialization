# What is Face Recognition?

 - Face recognition is a biometric technology that identifies or verifies a person's identity using their facial features.
 - It involves analyzing and comparing patterns based on a person's facial contours and structure.
 - Face recognition is widely used in various applications, including security systems, social media tagging, and etc.
 - Terminology in face recognition:
    - **Face Verification:** A one-to-one comparison to determine if the face matches a claimed identity.
    - **Face Identification:** A one-to-many comparison to identify an individual from a database of faces.

**Key Aspects of Face Recognition**

 - **Face Detection**: Identifying and locating a face in an image or video stream.
 - **Feature Extraction**: Analyzing specific characteristics of the face, such as the distance between the eyes, nose shape, or jawline, to create a unique facial signature or template.
 - **Face Matching**: Comparing the extracted facial features with a stored database to either verify the identity (face verification) or identify the individual among multiple people (face identification).

# One Shot Learning

 - One-shot learning enables a model to recognize a class or category after being exposed to just a single example.
 - Unlike traditional methods that require large datasets, one-shot learning generalizes from minimal data, making it ideal for tasks like face recognition, where only one image per person may be available.
 - So, instead of direct classification, the model learns a similarity function $ d $ that measures the difference between two images.
 - This approach allows the model to recognize new individuals by simply adding their image to the database, without the need for retraining.
 - The function $ d $ effectively addresses the challenges of one-shot learning.

**How One-Shot Learning Works**

 - Compare the new image with all images in the database using the function $ d $.
 - Function $ d $ output a threshold value $ \tau $:
    - $ d(\text{image1}, \text{image2}) < \tau $: Predict that the images are of the same person.
    - $ d(\text{image1}, \text{image2}) > \tau $: Predict that the images are of different people.
 - If all comparisons yield large $ d $ values, conclude that the person is not in the database.

# Siamese Network

 - A Siamese network consists of two identical convolutional neural networks (CNNs) that share the same parameters.
 - The same input image is passed through both networks to produce a feature vector (encoding) for each image.
 - The output of the CNN is a 128-dimensional feature vector, denoted as $ f(x) $, which encodes the input image $ x $.
 - For two images, $ x_1 $ and $ x_2 $, their encodings are represented as $ f(x_1) $ and $ f(x_2) $, respectively.
 - The similarity between the two images is measured by calculating the norm of the difference between their encodings: 
 $$ d(x_1, x_2) = \lVert f(x_1) - f(x_2) \rVert $$
 - A small distance indicates that the images are of the same person, while a large distance suggests they are of different people.
 - The Siamese network is trained to ensure that the distance $ d(x_i, x_j) $ is small when the images $ x_i $ and $ x_j $ are of the same person.
 - Conversely, the distance should be large when the images are of different people.

# Triplet Loss

 - The goal of the Tripe Loss function is to learn the neural network parameters that yield good encodings for face images.
 - The process involves three images:
    - **Anchor ($A$)**: The reference image.
    - **Positive ($P$)**: An image of the same person as the anchor.
    - **Negative ($N$)**: An image of a different person.
 - We have to ensure the distance between the $A$ and $P$ encoding is smaller than the distance between the $A$ and $N$ encoding.
 - Note that selecting challenging triplets ensures the gradient descent procedure is effective, avoiding trivial solutions.

**Formalizing the Triplet Loss Function**

 - The difference between the encodings is defined as: 
    $$ d(A, P) = \|f(A) - f(P)\|^2 $$
    $$ d(A, N) = \|f(A) - f(N)\|^2 $$
 - The basic inequality for triplet loss: 
    $$ d(A, P) \leq d(A, N) - \alpha $$
    where $ \alpha $ is a margin parameter ensuring the difference between distances is significant.
 - Reformulated inequality:
    $$ d(A, P) - d(A, N) + \alpha \leq 0 $$
 - Final loss function:
    $$ L(A, P, N) = \max(d(A, P) - d(A, N) + \alpha, 0) $$
    - If the inequality is satisfied, the loss is zero.
    - If not, the loss is positive, and gradient descent will work to minimize this loss.

# Binary Classification Approach 

 - This approach offers an alternative to triplet loss by framing face verification as a binary classification problem, using Siamese networks and logistic regression.
 - The embeddings from Anchor ($A$) and Positive ($P$) are fed into a logistic regression unit to classify whether the two images represent the same person or not.

**Formalizing the Binary Classification**

 - Compute the element-wise absolute difference between the embeddings:
    $$ \text{Feature}_k = |f(x_i)_k - f(x_j)_k| $$
    where $ f(x_i) $ and $ f(x_j) $ are the embeddings of images $ x_i $ and $ x_j $, respectively, and $ k $ denotes the $ k $-th component of the embedding.
 - These differences serve as features for the logistic regression unit.
 - A sigmoid function applied to the features to produce a probability:
    - 1 if the images are of the same person.
    - 0 if the images are of different people.
